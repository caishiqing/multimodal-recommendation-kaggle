from transformers import BertConfig, BertTokenizer
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

from model import RecModel, build_model
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
        rec_model = RecModel(self.config, self.item_model, self.user_model)

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

        # Build checkpoint and train model
        top_k = self.config.get('top_k', 12)
        checkpoint = Checkpoint(
            save_path, data,
            top_k=kwargs.get('top_k', top_k),
            max_history_length=self.config.get('max_history_length', 50),
            batch_size=kwargs.get('infer_batch_size', 256)
        )
        rec_model.fit(dataset, epochs=kwargs.get('epochs', 10), callbacks=[checkpoint])


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
        user_test_wrapper = self.data.test_wrapper
        profile = self.data.profile_data

        item_vectors = []
        with tqdm(total=len(self.data.items) // self.batch_size, desc='Compute items') as pbar:
            for batch_inputs in self.data.item_dataset(self.batch_size):
                pbar.update()
                batch_vector = self.model.item_model(batch_inputs).numpy()
                item_vectors.append(batch_vector)

        item_vectors = np.vstack(item_vectors)
        item_vectors[-1] *= 0

        predictions = []
        with tqdm(total=len(user_test_wrapper) // self.batch_size, desc='Compute recommendations') as pbar:
            for i in range(0, len(user_test_wrapper), self.batch_size):
                pbar.update()
                j = min(i + self.batch_size, len(user_test_wrapper))
                batch_profile = profile[user_test_wrapper.user_indices[i:j]]
                trans_indices = user_test_wrapper.trans_indices[i:j]
                trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
                    trans_indices, maxlen=self.max_history_length, value=-1
                ).reshape([-1])
                context = np.asarray(
                    self.data.trans['context'][trans_indices].to_list(),
                    dtype=np.int32
                ).reshape([j-i, self.max_history_length, -1])
                items = item_vectors[trans_indices].reshape([j-i, self.max_history_length, -1])
                user_vectors = self.model.user_model.predict(
                    {
                        'profile': batch_profile,
                        'context': context,
                        'items': items
                    }
                )
                # Apply dot similarity
                score = np.matmul(user_vectors, item_vectors.T)
                # Exclude interacted items in history
                for i, gt in enumerate(user_test_wrapper.ground_truth[i:j]):
                    score[i, gt] -= 1e-5

                # Cut off topk most related items
                predictions.append(np.argsort(-score, axis=-1)[:, :self.top_k])

        predictions = np.vstack(predictions)
        map = MAP(self.top_k)(self.data.test_wrapper.ground_truth, predictions)
        logs[self.monitor] = map
        super(Checkpoint, self).on_epoch_end(epoch, logs)

        print(self.monitor, ': ', map)
