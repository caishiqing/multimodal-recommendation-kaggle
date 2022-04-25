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
        item_data = {
            'info': data.info_data,
            'desc': data.desc_data,
            'image': data.image_data
        }
        print(item_data['info'].device)
        print(item_data['desc'].device)
        print(item_data['image'].device)
        print(self.item_model.trainable_weights[0].device)
        rec_model = RecModel(self.config, self.item_model, self.user_model, item_data)

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
            initial_learning_rate=kwargs.get('learning_rate', 1e-4)
        )
        rec_model.compile(
            optimizer=optimizer,
            margin=kwargs.get('margin', 0.0),
            gamma=kwargs.get('gamma', 1.0)
        )
        self.item_model.summary()
        self.user_model.summary()

        # Build checkpoint and train model
        top_k = self.config.get('top_k', 12)
        checkpoint = Checkpoint(
            save_path, data,
            top_k=kwargs.get('top_k', top_k),
            max_history_length=self.config.get('max_history_length', 50),
            batch_size=kwargs.get('infer_batch_size', 256),
            skip_used_items=kwargs.get('skip_used_items', False)
        )
        rec_model.fit(
            dataset, epochs=kwargs.get('epochs', 10),
            steps_per_epoch=kwargs.get('steps_per_epoch'),
            callbacks=[checkpoint]
        )

    def infer(self, data: RecData,
              batch_size: int = 128,
              skip_used_items=False,
              top_k=10,
              verbose=0):

        infer_wrapper = self.data.infer_wrapper
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            infer_wrapper.trans_indices, maxlen=self.max_history_length, value=-1
        ).reshape([-1])
        profile = self.data.user_data['profile'][infer_wrapper.user_indices]
        context = self.data.trans_data['context'][trans_indices].reshape(
            [len(profile), self.max_history_length, -1])
        item_indices = np.asarray(
            self.data.trans.iloc[trans_indices]['item'], np.int32
        ).reshape([len(profile), -1])

        item_vectors = self.item_model.predict(data.item_data,
                                               batch_size=self.batch_size,
                                               verbose=verbose)
        item_vectors = tf.identity(item_vectors)

        infer_model = RecInfer(
            self.model.user_model,
            item_vectors,
            top_k=self.top_k,
            skip_used_items=skip_used_items
        )
        infer_inputs = {'profile': profile, 'context': context, 'item_indices': item_indices}
        predictions = infer_model.predict(infer_inputs,
                                          batch_size=batch_size,
                                          verbose=verbose)

        return predictions


class Checkpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath: str,
                 data: RecData,
                 top_k: int = 12,
                 max_history_length: int = 50,
                 batch_size: int = 256,
                 skip_used_items: bool = False,
                 **kwargs):

        monitor = f'MAP@{top_k}'
        super(Checkpoint, self).__init__(
            filepath,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            **kwargs
        )
        self.data = data
        self.max_history_length = max_history_length
        self.batch_size = batch_size
        self.top_k = top_k
        self.skip_used_items = skip_used_items

    def on_epoch_end(self, epoch, logs):
        test_wrapper = self.data.test_wrapper
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            test_wrapper.trans_indices, maxlen=self.max_history_length, value=-1
        ).reshape([-1])
        item_indices = np.asarray(
            self.data.trans.iloc[trans_indices]['item'],
            np.int32).reshape([len(test_wrapper), -1])
        ground_truth = tf.keras.preprocessing.sequence.pad_sequences(
            test_wrapper.ground_truth, maxlen=self.top_k,
            padding='post', truncating='post', value=-1
        )

        item_vectors = self.model.item_model.predict(
            self.model.item_data, batch_size=self.batch_size)
        item_vectors = tf.identity(item_vectors)

        profile = self.data.user_data['profile'][test_wrapper.user_indices]
        context = self.data.trans_data['context'][trans_indices].reshape(
            [len(test_wrapper), self.max_history_length, -1])

        infer_model = RecInfer(
            self.model.user_model,
            item_vectors,
            top_k=self.top_k,
            skip_used_items=self.skip_used_items
        )
        infer_model.compile(metrics=MAP(self.top_k))
        infer_inputs = {'profile': profile, 'context': context, 'item_indices': item_indices}
        _, map_score = infer_model.evaluate(infer_inputs, ground_truth)

        logs[self.monitor] = map_score
        super(Checkpoint, self).on_epoch_end(epoch, logs)
        print('map@{}: {:.4}'.format(self.top_k, map_score))
