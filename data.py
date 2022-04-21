from transformers import BertTokenizer, BertConfig
from collections import OrderedDict, namedtuple
from typing import List, Union
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import base64
import json
import os


class DataWrapper(object):
    def __init__(self):
        self.user_indices = []
        self.trans_indices = []
        self.ground_truth = []
        self.index = dict()

    def append(self, user_indice: int,
               trans_indices: List[int],
               ground_truth_indices: List[int] = None):

        assert (user_indice not in self.index,
                f'UserIndice {user_indice} is already in, please use `set_value` function!')

        self.user_indices.append(user_indice)
        self.trans_indices.append(trans_indices)
        if ground_truth_indices is not None:
            self.ground_truth.append(ground_truth_indices)

        self.index[user_indice] = len(self) - 1

    def set_value(self,
                  user_indice: int,
                  trans_indices: List[int],
                  ground_truth: List[int] = None):

        index = self.index[user_indice]
        self.user_indices[index] = user_indice
        self.trans_indices[index] = trans_indices
        if ground_truth is not None:
            self.ground_truth[index] = ground_truth

    def shuffle(self):
        indices = list(range(len(self)))
        self.user_indices = [self.user_indices[i] for i in indices]
        self.trans_indices = [self.trans_indices[i] for i in indices]
        if self.ground_truth:
            self.ground_truth = [self.ground_truth[i] for i in indices]

    def __len__(self):
        return len(self.user_indices)


class RecData(object):
    _sys_fields = (
        'id', 'desc', 'info', 'image', 'tfrecord',
        'profile', 'context', 'user', 'item', 'trans'
    )

    def __init__(self,
                 items: pd.DataFrame,
                 users: pd.DataFrame,
                 trans: pd.DataFrame,
                 config: Union[dict, BertConfig],
                 feature_path: str = None,
                 resize_image: bool = False):

        assert 'id' in items and 'id' in users
        assert 'item' in trans and 'user' in trans

        self.item_feature_dict = OrderedDict()
        self.user_feature_dict = OrderedDict()
        self.trans_feature_dict = OrderedDict()
        self.train_wrapper = DataWrapper()
        self.test_wrapper = DataWrapper()

        self.config = build_config(config)
        self.resize_image = resize_image

        self.items = items
        self.users = users
        self.trans = trans
        self.items.reset_index(drop=True, inplace=True)
        self.users.reset_index(drop=True, inplace=True)
        self.trans.reset_index(drop=True, inplace=True)

        # items and users use reseted index
        self.item_index_map = OrderedDict([(id, i) for i, id in enumerate(self.items['id'])])
        self.user_index_map = OrderedDict([(id, i) for i, id in enumerate(self.users['id'])])
        self.trans['item'] = self.trans['item'].map(self.item_index_map)
        self.trans['user'] = self.trans['user'].map(self.user_index_map)

        # load/learn features maps
        if feature_path is not None:
            self.load_feature_dict(feature_path)
        else:
            self._learn_feature_dict()

    def prepare_features(self, tokenizer: BertTokenizer):
        if not self._processed:
            print('Process item features ...', end='')
            # length + 1 for padding
            for key, feat_map in self.item_feature_dict.items():
                self.items[key] = self.items[key].map(feat_map)

            self.items['desc'] = tokenizer(
                self.items['desc'].to_list(),
                max_length=self.config.max_desc_length,
                truncation=True,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']

            # self.image_data = list(tf.data.Dataset.from_tensor_slices(
            #     self.items.pop('image').map(base64.b64decode)).map(
            #     tf.image.decode_jpeg, tf.data.experimental.AUTOTUNE).batch(len(self.items)))[0].numpy()
            self.image_data = tf.map_fn(
                tf.image.decode_jpeg, self.items.pop('image').map(base64.b64decode),
                dtype=tf.uint8).numpy()
            print('Done!')

            print('Process user features ...', end='')
            for key, feat_map in self.user_feature_dict.items():
                self.users[key] = self.users[key].map(feat_map)
            print('Done!')

            print('Process transaction features ...', end='')
            for key, feat_map in self.trans_feature_dict.items():
                self.trans[key] = self.trans[key].map(feat_map)
            print('Done!')
        else:
            print("Features are aleady prepared.")

        self._padding()

    @property
    def _processed(self):
        flag = isinstance(self.items['desc'][0], list)
        return flag

    def _padding(self):
        padding = {col: 0 for col in self.items}
        padding['id'] = -1
        padding['desc'] = [0]
        self.items.loc[-1] = padding
        self.image_data = np.vstack([self.image_data, np.zeros((1,)+self.image_data.shape[1:], np.uint8)])

        padding = {col: 0 for col in self.users}
        padding['id'] = -1
        self.users.loc[-1] = padding

        padding = {col: 0 for col in self.trans}
        padding['id'] = -1
        self.trans.loc[-1] = padding

    @property
    def _padded(self):
        return -1 in self.items.index and -1 in self.users.index and -1 in self.trans.index

    def prepare_train(self, test_users: list = None):
        if test_users is not None:
            test_users = [self.user_index_map[user] for user in test_users]
            test_users = set(test_users)

        with tqdm(total=len(self.trans['user'].unique()), desc='Process training data') as pbar:
            for user_idx, df in self.trans.groupby('user'):
                pbar.update()
                trans_indices = df.index.to_list()
                item_indices = df['item'].to_list()
                if len(trans_indices) < self.config.max_history_length or (test_users is not None and user_idx in test_users):
                    # test sample
                    if len(trans_indices) == 1:
                        # no transactions, only use profile
                        self.test_wrapper.append(user_idx, [], item_indices)
                    elif len(df) < self.config.top_k:
                        self.test_wrapper.append(user_idx, trans_indices[:1], item_indices[1:])
                    else:
                        self.test_wrapper.append(
                            user_idx,
                            trans_indices[:-self.config.top_k],
                            item_indices[-self.config.top_k:]
                        )
                else:
                    # train sample
                    cut_offset = max(len(trans_indices)-self.config.top_k, self.config.max_history_length)
                    self.train_wrapper.append(user_idx, trans_indices[:cut_offset])
                    if cut_offset < len(trans_indices):
                        # cut off for test
                        self.test_wrapper.append(user_idx, trans_indices[:cut_offset], item_indices[cut_offset:])

        # shuffle train samples
        self.train_wrapper.shuffle()

        print('Train samples: {}'.format(len(self.train_wrapper)))
        print('Test samples: {}'.format(len(self.test_wrapper)))

    @property
    def item_data(self):
        info = np.asarray(self.items[self.item_feature_dict.keys()], np.int32)
        desc = tf.keras.preprocessing.sequence.pad_sequences(
            self.items['desc'], maxlen=self.config.max_desc_length, dtype=np.int32,
            padding='post', truncating='post', value=0
        )
        return {
            'info': info,
            'desc': desc,
            'image': self.image_data
        }

    @property
    def profile_data(self):
        return np.asarray(self.users[self.user_feature_dict.keys()], np.int32)

    @property
    def context_data(self):
        return np.asarray(self.trans[self.trans_feature_dict.keys()], np.int32)

    @property
    def infer_wrapper(self):
        wrapper = DataWrapper()
        for user_idx, df in self.trans.groupby('user'):
            trans_indices = df.index.to_list()
            wrapper.append(user_idx, trans_indices)

        return wrapper

    @property
    def info_size(self):
        size = []
        for feat, feat_map in self.item_feature_dict.items():
            size.append(len(feat_map))

        return size

    @property
    def profile_size(self):
        size = []
        for feat, feat_map in self.user_feature_dict.items():
            size.append(len(feat_map))

        return size

    @property
    def context_size(self):
        size = []
        for feat, feat_map in self.trans_feature_dict.items():
            size.append(len(feat_map))

        return size

    def _learn_feature_dict(self):
        for col in self.items.columns:
            if col in self._sys_fields:
                continue
            vals = set(self.items[col])
            self.item_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(sorted(vals))])

        for col in self.users.columns:
            if col in self._sys_fields:
                continue
            vals = set(self.users[col])
            self.user_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(sorted(vals))])

        for col in self.trans.columns:
            if col in self._sys_fields:
                continue
            vals = set(self.trans[col])
            self.trans_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(sorted(vals))])

        self._display_feature_info()

    def _display_feature_info(self):
        info = []
        for feat, feat_map in self.item_feature_dict.items():
            info.append({'subject': 'item', 'feature': feat, 'size': len(feat_map)})
        for feat, feat_map in self.user_feature_dict.items():
            info.append({'subject': 'user', 'feature': feat, 'size': len(feat_map)})
        for feat, feat_map in self.trans_feature_dict.items():
            info.append({'subject': 'trans', 'feature': feat, 'size': len(feat_map)})

        info = pd.DataFrame(info, index=None)
        print(info)

    def train_dataset(self, batch_size: int = 8):
        assert self._processed and self._padded
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            self.train_wrapper.trans_indices, maxlen=self.config.max_history_length,
            padding='pre', truncating='pre', value=-1
        ).reshape([-1])
        item_indices = self.trans.iloc[trans_indices]['item']

        profile = self.profile_data[self.train_wrapper.user_indices]
        context = self.context_data[trans_indices].reshape(
            [-1, self.config.max_history_length, len(self.context_size)])
        items = np.asarray(item_indices, np.int32).reshape([-1, self.config.max_history_length])

        dataset = tf.data.Dataset.from_tensor_slices(
            {
                'profile': profile,
                'context': context,
                'items': items
            }
        ).shuffle(2*batch_size).batch(batch_size)

        return dataset

    def save_feature_dict(self, save_dir: str):
        # Save feature dict to direction
        with open(os.path.join(save_dir, 'item_feature_dict.json'), 'w', encoding='utf8') as fp:
            json.dump(self.item_feature_dict, fp)
        with open(os.path.join(save_dir, 'user_feature_dict.json'), 'w', encoding='utf8') as fp:
            json.dump(self.user_feature_dict, fp)
        with open(os.path.join(save_dir, 'trans_feature_dict.json'), 'w', encoding='utf8') as fp:
            json.dump(self.trans_feature_dict, fp)

    def load_feature_dict(self, load_dir: str):
        # Load feature dict from direction
        with open(os.path.join(load_dir, 'item_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.item_feature_dict = OrderedDict(json.load(fp))
        with open(os.path.join(load_dir, 'user_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.user_feature_dict = OrderedDict(json.load(fp))
        with open(os.path.join(load_dir, 'trans_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.trans_feature_dict = OrderedDict(json.load(fp))
        self._display_feature_info()


def build_config(config):
    if isinstance(config, dict):
        Config = namedtuple('Config', config.keys())
        config = Config(**config)

    return config
