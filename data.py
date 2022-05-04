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

        self.config = config
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

        self.item_data = None
        self.user_data = None
        self.trans_data = None

    def prepare_features(self, tokenizer: BertTokenizer):
        if not self._processed:
            print('Process item features ...', end='')
            # length + 1 for padding
            info = np.zeros((len(self.items), len(self.info_size)), dtype=np.int32)
            for i, (key, feat_map) in enumerate(self.item_feature_dict.items()):
                info[:, i] = self.items.pop(key).map(feat_map)

            desc = tokenizer(
                self.items.pop('desc').to_list(),
                max_length=self.config.get('max_desc_length', 8),
                truncation=True,
                padding='max_length',
                return_attention_mask=False,
                return_token_type_ids=False,
                return_tensors='np'
            )['input_ids']

            def _decode_image(img_bytes):
                img = tf.image.decode_image(img_bytes, expand_animations=False)
                img = tf.image.resize(img, size=(self.config['image_height'], self.config['image_width']))
                return img.numpy()

            image = np.asarray([_decode_image(img) for img in self.items.pop('image').map(base64.b64decode)], np.uint8)
            self.item_data = {'info': info, 'desc': desc, 'image': image}
            print('Done!')

            print('Process user features ...', end='')
            profile = np.zeros((len(self.users), len(self.profile_size)), dtype=np.int32)
            for i, (key, feat_map) in enumerate(self.user_feature_dict.items()):
                profile[:, i] = self.users.pop(key).map(feat_map)

            self.user_data = {'profile': profile}
            print('Done!')

            print('Process transaction features ...', end='')
            context = np.zeros((len(self.trans), len(self.context_size)), dtype=np.int32)
            for i, (key, feat_map) in enumerate(self.trans_feature_dict.items()):
                context[:, i] = self.trans.pop(key).map(feat_map)

            self.trans_data = {'context': context}
            self.trans.loc[-1] = {'item': -1, 'user': -1}  # for padding indice
            print('Done!')
        else:
            print("Features are aleady prepared.")

    @property
    def _processed(self):
        flag = self.item_data is not None
        flag &= self.user_data is not None
        flag &= self.trans_data is not None
        return flag

    def prepare_train(self, test_users: list = None):
        if test_users is not None:
            test_users = [self.user_index_map[user] for user in test_users]
            test_users = set(test_users)

        with tqdm(total=len(self.trans['user'].unique()), desc='Process training data') as pbar:
            for user_idx, df in self.trans.groupby('user'):
                pbar.update()
                trans_indices = df.index.to_list()
                item_indices = df['item'].to_list()
                if len(trans_indices) < self.config.get('max_history_length', 32) or (test_users is not None and user_idx in test_users):
                    # test sample
                    if len(trans_indices) == 1:
                        # no transactions, only use profile
                        self.test_wrapper.append(user_idx, [], item_indices)
                    elif len(df) < self.config.get('top_k', 10):
                        self.test_wrapper.append(user_idx, trans_indices[:1], item_indices[1:])
                    else:
                        self.test_wrapper.append(
                            user_idx,
                            trans_indices[:-self.config.get('top_k', 10)],
                            item_indices[-self.config.get('top_k', 10):]
                        )
                else:
                    # train sample
                    cut_offset = max(len(trans_indices)-self.config.get('top_k', 10), self.config.get('max_history_length', 32))
                    self.train_wrapper.append(user_idx, trans_indices[:cut_offset])
                    if cut_offset < len(trans_indices):
                        # cut off for test
                        self.test_wrapper.append(user_idx, trans_indices[:cut_offset], item_indices[cut_offset:])

        # shuffle train samples
        self.train_wrapper.shuffle()

        print('Train samples: {}'.format(len(self.train_wrapper)))
        print('Test samples: {}'.format(len(self.test_wrapper)))

    @property
    def infer_wrapper(self):
        wrapper = DataWrapper()
        trans_data = dict(list(self.trans.groupby('user')))
        for i in self.users.index:
            trans_indices = trans_data.get(i)
            trans_indices = trans_indices.to_list() if trans_indices else []
            wrapper.append(i, trans_indices)

        return wrapper

    @property
    def info_size(self):
        size = []
        for _, feat_map in self.item_feature_dict.items():
            size.append(len(feat_map))

        return size

    @property
    def profile_size(self):
        size = []
        for _, feat_map in self.user_feature_dict.items():
            size.append(len(feat_map))

        return size

    @property
    def context_size(self):
        size = []
        for _, feat_map in self.trans_feature_dict.items():
            size.append(len(feat_map))

        return size

    def _learn_feature_dict(self):
        info_features = sorted(set(self.items.columns) - set(self._sys_fields))
        if self.config.get('use_item_id'):
            info_features.append('id')
        for col in info_features:
            vals = set(self.items[col])
            self.item_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(sorted(vals))])

        profile_features = sorted(set(self.users.columns) - set(self._sys_fields))
        if self.config.get('use_user_id'):
            profile_features.append('id')
        for col in profile_features:
            vals = set(self.users[col])
            self.user_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(sorted(vals))])

        context_features = sorted(set(self.trans.columns) - set(self._sys_fields))
        for col in context_features:
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
        assert self._processed
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            self.train_wrapper.trans_indices, maxlen=self.config.get('max_history_length', 32),
            padding='pre', truncating='pre', value=-1
        ).reshape([-1])
        item_indices = self.trans.iloc[trans_indices]['item']

        data = {
            'profile': self.user_data['profile'][self.train_wrapper.user_indices],
            'context': self.trans_data['context'][trans_indices].reshape(
                [len(self.train_wrapper), self.config.get('max_history_length', 32), -1]),
            'items': np.asarray(item_indices, np.int32).reshape([-1, self.config.get('max_history_length', 32)])
        }
        dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(2*batch_size).batch(batch_size, drop_remainder=True)
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
