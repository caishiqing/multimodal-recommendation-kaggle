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
            'info': tf.identity(data.info_data),
            'desc': tf.identity(data.desc_data),
            'image': tf.identity(data.image_data)
        }
        print(item_data['info'].device)
        print(item_data['desc'].device)
        print(item_data['image'].device)
        print(self.item_model.trainable_weights[0].device)
        rec_model = RecModel(self.config,
                             self.item_model,
                             self.user_model,
                             item_data)

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
            batch_size=kwargs.get('infer_batch_size', 256)
        )
        rec_model.fit(
            dataset, epochs=kwargs.get('epochs', 10),
            steps_per_epoch=kwargs.get('steps_per_epoch'),
            callbacks=[checkpoint]
        )

    def infer(self, data: RecData, batch_size: int = 128, top_k=10):
        item_vectors = self.item_model.predict(
            data.item_data, batch_size=batch_size, verbose=1)
        item_vectors[-1] *= 0  # for padding
        user_vectors = self.user_model.infer_initial_state(
            data.profile_data, batch_size=batch_size, verbose=1)

        infer_wrapper = data.infer_wrapper
        context = data.context_data
        with tqdm(total=len(infer_wrapper)//batch_size, desc='Computing recommendations') as pbar:
            for i in range(0, len(infer_wrapper), batch_size):
                pbar.update()
                j = min(i + self.batch_size, len(infer_wrapper))
                user_indices = infer_wrapper.user_indices[i:j]
                trans_indices = infer_wrapper.trans_indices[i:j]
                trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
                    trans_indices, maxlen=self.config['max_history_length'], value=-1
                ).reshape([-1])
                item_indices = self.data.trans['item'][trans_indices]
                batch_context = context[trans_indices].reshape([j-i, self.max_history_length, -1])
                batch_item_vectors = item_vectors[item_indices].reshape([j-i, self.max_history_length, -1])
                batch_init_state = user_vectors[user_indices]
                _, batch_user_vectors = self.model.user_model.predict(
                    {
                        'init_state': batch_init_state,
                        'context': batch_context,
                        'items': batch_item_vectors
                    },
                    batch_size=batch_size
                )
                # update user state vectors
                user_vectors[user_indices] = batch_user_vectors

            predictions = []
            for i in range(0, len(data.users), batch_size):
                j = min(i + self.batch_size, len(data.users))
                batch_user_vectors = user_vectors[i:j]
                # Apply dot similarity
                score = np.matmul(batch_user_vectors, item_vectors[:-1].T)
                # Exclude interacted items in history
                for user_indice in range(i, j):
                    if user_indice not in infer_wrapper.index:
                        continue
                    k = infer_wrapper.index[user_indice]
                    trans_indices = infer_wrapper.trans_indices[k]
                    used_items = data.trans['item'][trans_indices]
                    score[i, used_items] -= 1e-5

                # Cut off topk most related items
                predictions.append(np.argsort(-score, axis=1)[:top_k])

            predictions = np.vstack(predictions)
            return predictions


class Checkpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath: str,
                 data: RecData,
                 top_k: int = 12,
                 max_history_length: int = 50,
                 batch_size: int = 256,
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

    def on_epoch_end(self, epoch, logs):
        test_wrapper = self.data.test_wrapper
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            test_wrapper.trans_indices, maxlen=self.max_history_length, value=-1
        ).reshape([-1])
        profile = self.data.profile_data[test_wrapper.user_indices]
        context = self.data.context_data[trans_indices].reshape([len(profile), -1])
        item_indices = np.asarray(self.data.trans['item'][trans_indices], np.int32).reshape([len(profile), -1])
        ground_truth = tf.keras.preprocessing.sequence.pad_sequences(
            test_wrapper.ground_truth, maxlen=self.top_k,
            padding='post', truncating='post', value=-1
        )

        item_vectors = self.model.item_model.predict(
            self.model.item_data, batch_size=self.batch_size, verbose=1)
        # last item for padding
        item_vectors[-1] *= 0
        item_vectors = tf.identity(item_vectors)

        infer_model = RecInfer(self.model.user_model, item_vectors, top_k=self.top_k)
        infer_model.compile(metrics=MAP(self.top_k))
        infer_inputs = {'profile': profile, 'context': context, 'item_indices': item_indices}
        _, map_score = infer_model.evaluate(infer_inputs, ground_truth)

        logs[self.monitor] = map_score
        super(Checkpoint, self).on_epoch_end(epoch, logs)
        print('map@{}: {:.4}'.format(self.top_k, map_score))
