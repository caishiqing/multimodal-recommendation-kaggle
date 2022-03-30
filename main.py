from errno import ECOMM
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
          train_data: RecData,
          valid_data: RecData,
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
    model_config = BertConfig.from_pretrained(config['bert_path'])
    model_config.update(config)
    train_data.learn_feature_dict()
    valid_data.load_feature_dict(train_data)
    train_dataloader = DataLoader(
        model_config, train_data.items, train_data.users, train_data.transactions,
        train_data.item_feature_dict, train_data.user_feature_dict, tokenizer
    )
    valid_dataloader = DataLoader(
        model_config, valid_data.items, valid_data.users, valid_data.transactions,
        valid_data.item_feature_dict, valid_data.user_feature_dict, tokenizer
    )
    train_dataset = train_dataloader.train_dataset(batch_size)
    valid_dataset = valid_dataloader.train_dataset(batch_size)

    # Save model config and feature config files
    model_config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, 'item_feature_dict.json'), 'w', encoding='utf8') as fp:
        json.dump(train_data.item_feature_dict, fp, indent=2)
    with open(os.path.join(save_path, 'user_feature_dict.json'), 'w', encoding='utf8') as fp:
        json.dump(train_data.user_feature_dict, fp, indent=2)

    # Build and train model
    with strategy.scope():
        model = build_train_model(**config)
        total_steps = epochs * len(train_data.transactions) // batch_size
        optimizer = AdamWarmup(
            warmup_steps=int(total_steps * warmup_proportion),
            decay_steps=total_steps - int(total_steps * warmup_proportion),
            initial_learning_rate=learning_rate
        )
        model.compile(optimizer=optimizer)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_path, 'model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        )
        model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            callbacks=[checkpoint]
        )
