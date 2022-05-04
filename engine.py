from transformers import BertConfig, BertTokenizer
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

from model import RecModel, build_model, RecInfer
from optimizer import AdamWarmup
from data import RecData
from evaluate import MAP


class RecEngine:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.item_model, self.user_model = build_model(self.config)

    @classmethod
    def from_pretrained(cls, model_path: str):
        model_config = BertConfig.from_pretrained(model_path)
        config = model_config.to_dict()
        config['bert_path'] = model_path
        config['image_weights'] = None
        engine = cls(config)

        engine.item_model.load_weights(os.path.join(model_path, 'item.h5'))
        engine.user_model.load_weights(os.path.join(model_path, 'user.h5'))
        return engine

    def train(self, data: RecData,
              test_users: list = None,
              save_path: str = './model',
              **kwargs):

        os.makedirs(save_path, exist_ok=True)
        batch_size = kwargs.get('batch_size', 32)

        data.prepare_features(self.tokenizer)
        data.prepare_train(test_users)
        dataset = data.train_dataset(batch_size)
        rec_model = RecModel(self.config,
                             self.item_model,
                             self.user_model,
                             data.item_data)

        # Save files related to model
        model_config = BertConfig.from_pretrained(self.config.get('bert_path', 'bert-base-uncased'))
        model_config.update(self.config)
        model_config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        data.save_feature_dict(save_path)

        # Compile model with optimizer
        total_steps = kwargs.get('epochs', 10) * len(data.train_wrapper) // kwargs.get('batch_size', 32)
        optimizer = AdamWarmup(
            warmup_steps=int(total_steps * kwargs.get('warmup_proportion', 0.1)),
            decay_steps=total_steps - int(total_steps * kwargs.get('warmup_proportion', 0.1)),
            initial_learning_rate=kwargs.get('learning_rate', 1e-4),
            lr_multiply=kwargs.get('lr_multiply')
        )
        rec_model.compile(
            optimizer=optimizer,
            margin=kwargs.get('margin', 0.0),
            gamma=kwargs.get('gamma', 1.0)
        )

        self.item_model.summary()
        self.user_model.summary()

        # Build checkpoint and train model
        with tf.device(self.item_model.trainable_weights[0].device):
            checkpoint = Checkpoint(
                save_path, data, self.config,
                batch_size=kwargs.get('infer_batch_size', 256),
                skip_used_items=kwargs.get('skip_used_items', False),
                verbose=kwargs.get('verbose', 1)
            )
        rec_model.fit(
            dataset, batch_size=batch_size,
            epochs=kwargs.get('epochs', 10),
            steps_per_epoch=kwargs.get('steps_per_epoch'),
            callbacks=[checkpoint],
            verbose=kwargs.get('verbose', 1)
        )

    def infer(self, data: RecData,
              batch_size: int = 128,
              skip_used_items=False,
              top_k=10,
              verbose=0):

        data.prepare_features(self.tokenizer)
        infer_wrapper = data.infer_wrapper
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            infer_wrapper.trans_indices, maxlen=self.config.get('max_history_length', 50), value=-1
        ).reshape([-1])
        profile = data.user_data['profile'][infer_wrapper.user_indices]
        context = data.trans_data['context'][trans_indices].reshape(
            [len(profile), self.config.get('max_history_length', 50), -1])
        item_indices = np.asarray(
            data.trans.iloc[trans_indices]['item'], np.int32
        ).reshape([len(profile), -1])

        infer_model = RecInfer(self.user_model,
                               skip_used_items=skip_used_items,
                               max_history_length=self.config['max_history_length'],
                               profile_dim=len(self.config['profile_size']),
                               context_dim=len(self.config['context_size']),
                               num_items=len(data.items),
                               embed_dim=self.config['embed_dim'],
                               top_k=top_k)

        item_vectors = self.item_model.predict(data.item_data,
                                               batch_size=batch_size,
                                               verbose=verbose)
        infer_model.set_item_vectors(item_vectors)
        infer_inputs = {'profile': profile, 'context': context, 'item_indices': item_indices}
        predictions = infer_model.predict(infer_inputs, batch_size=batch_size, verbose=verbose)

        return predictions


class Checkpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath: str,
                 data: RecData,
                 config: dict,
                 batch_size: int = 256,
                 skip_used_items: bool = False,
                 verbose: int = 1,
                 **kwargs):

        top_k = config.get('top_k', 10)
        monitor = 'MAP@{}'.format(top_k)
        super(Checkpoint, self).__init__(
            filepath,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            **kwargs
        )
        self.data = data
        self.config = config
        self.batch_size = batch_size
        self.skip_used_items = skip_used_items
        self.verbose = verbose
        self.top_k = top_k
        self.max_history_length = config.get('max_history_length', 32)
        self.profile_dim = len(config['profile_size'])
        self.context_dim = len(config['context_size'])

        test_wrapper = self.data.test_wrapper
        # use latest history transactions for observation
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            test_wrapper.trans_indices, maxlen=self.max_history_length,
            padding='pre', truncating='pre', value=-1
        ).reshape([-1])
        item_indices = np.asarray(
            self.data.trans.iloc[trans_indices]['item'],
            np.int32).reshape([len(test_wrapper), -1])
        # use earlest future transactions for forecasting
        ground_truth = tf.keras.preprocessing.sequence.pad_sequences(
            test_wrapper.ground_truth, maxlen=self.top_k,
            padding='post', truncating='post', value=-1
        )
        profile = self.data.user_data['profile'][test_wrapper.user_indices]
        context = self.data.trans_data['context'][trans_indices].reshape(
            [len(test_wrapper), self.max_history_length, -1])
        infer_inputs = {
            'profile': profile,
            'context': context,
            'item_indices': item_indices
        }
        self.eval_data = tf.data.Dataset.from_tensor_slices(
            (infer_inputs, ground_truth)).batch(self.batch_size, drop_remainder=True)

    def set_model(self, model):
        super(Checkpoint, self).set_model(model)
        self.infer_model = RecInfer(self.model.user_model,
                                    skip_used_items=self.skip_used_items,
                                    max_history_length=self.max_history_length,
                                    profile_dim=self.profile_dim,
                                    context_dim=self.context_dim,
                                    num_items=self.model.item_data['info'].shape[0],
                                    embed_dim=self.config['embed_dim'],
                                    top_k=self.top_k)

        self.infer_model.compile(metrics=MAP(self.top_k))

    def on_epoch_end(self, epoch, logs):
        item_vectors = self.model.item_model.predict(self.model.item_data,
                                                     batch_size=self.batch_size,
                                                     verbose=self.verbose)

        self.infer_model.set_item_vectors(item_vectors)
        _, map_score = self.infer_model.evaluate(self.eval_data, verbose=self.verbose)

        logs[self.monitor] = map_score
        super(Checkpoint, self).on_epoch_end(epoch, logs)
        print('{}: {:.4}'.format(self.monitor, map_score))
