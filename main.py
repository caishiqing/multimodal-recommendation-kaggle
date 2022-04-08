from typing import List
from model import Item, Items, User, build_train_model
from transformers import BertTokenizer, BertConfig
from data import DataLoader, RecData
from optimizer import AdamWarmup
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os


def train(config: dict,
          data: RecData,
          test_users: list,
          save_path: str,
          epochs: int = 10,
          batch_size: int = 32,
          warmup_proportion: float = 0.1,
          learning_rate=1e-4,
          device_type: str = 'gpu'):

    # Build distribute strategy on gpu or tpu
    if device_type.lower() == 'tpu':
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            strategy = tf.distribute.get_strategy()
    elif device_type.lower() == 'gpu':
        strategy = tf.distribute.MirroredStrategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    # Process data
    tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
    data.process_features(tokenizer)
    data.process_transactions('train', test_users)
    data_loader = DataLoader(config, data)
    train_dataset = data_loader.train_dataset(batch_size)

    # Save model config and feature config files
    model_config = BertConfig.from_pretrained(config['bert_path'])
    model_config.update(config)
    model_config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    data.save_feature_dict(save_path)

    # Build and train model
    with strategy.scope():
        model, items_model, user_model = build_train_model(**config)
        total_steps = epochs * len(data.train) // batch_size
        optimizer = AdamWarmup(
            warmup_steps=int(total_steps * warmup_proportion),
            decay_steps=total_steps - int(total_steps * warmup_proportion),
            initial_learning_rate=learning_rate
        )
        model.compile(optimizer=optimizer)
        model_path = os.path.join(save_path, 'model.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        )
        model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[checkpoint]
        )

    # Save model weights file
    model.load_weights(model_path)
    items_model.save_weights(os.path.join(save_path, 'item.h5'))
    user_model.save_weights(os.path.join(save_path, 'user.h5'))
    os.remove(model_path)
