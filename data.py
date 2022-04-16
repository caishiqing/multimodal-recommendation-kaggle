import profile
from black import token
from transformers import BertTokenizer, BertConfig
from collections import OrderedDict, namedtuple
from typing import List, Union
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os


class DataWrapper(object):
    def __init__(self):
        self.user_indices = []
        self.trans_indices = []
        self.ground_truth = []

    def append(self, user_indice: int,
               trans_indices: List[int],
               ground_truth_indices: List[int] = None):

        self.user_indices.append(user_indice)
        self.trans_indices.append(trans_indices)
        if ground_truth_indices is not None:
            self.ground_truth.append(ground_truth_indices)

    def shuffle(self):
        indices = list(range(len(self)))
        self.user_indices = [self.user_indices[i] for i in indices]
        self.trans_indices = [self.trans_indices[i] for i in indices]
        if self.ground_truth:
            self.ground_truth = [self.ground_truth[i] for i in indices]

    def __len__(self):
        return len(self.user_indices)


class RecData(object):
    item_feature_dict = OrderedDict()
    user_feature_dict = OrderedDict()
    trans_feature_dict = OrderedDict()
    train_wrapper = DataWrapper()
    test_wrapper = DataWrapper()
    _padded = False
    _processed = False
    _sys_fields = ('id', 'desc', 'image', 'trans', 'user', 'item')

    def __init__(self,
                 items: pd.DataFrame,
                 users: pd.DataFrame,
                 trans: pd.DataFrame,
                 config: Union[dict, BertConfig],
                 feature_path: str = None,
                 resize_image: bool = False):

        assert 'id' in items and 'id' in users
        assert 'item' in trans and 'user' in trans
        self.config = build_config(config)
        self.resize_image = resize_image

        self.items = items.reset_index(drop=True)
        self.users = users.reset_index(drop=True)
        self.trans = trans

        # items and users use reseted index
        self.item_index_map = OrderedDict([(id, i) for i, id in enumerate(self.items['id'])])
        self.user_index_map = OrderedDict([(id, i) for i, id in enumerate(self.users['id'])])
        self.trans.reset_index(drop=True, inplace=True)
        self.trans['item'] = self.trans['item'].map(self.item_index_map)
        self.trans['user'] = self.trans['user'].map(self.user_index_map)

        # load/learn features maps
        if feature_path is not None:
            self.load_feature_dict(feature_path)
        else:
            self._learn_feature_dict()

    @property
    def include_image(self):
        return 'image' in self.items or 'image_path' in self.items

    @property
    def include_desc(self):
        return 'desc' in self.items

    def prepare_features(self, tokenizer: BertTokenizer = None):
        if not self._processed:
            self._process_item_features(tokenizer)
            self._process_user_features()
            self._process_transaction_features()
            self._processed = True

        self._display_feature_info()

    def prepare_train(self, test_users: list = None):
        if test_users is not None:
            test_users = [self.user_index_map[user] for user in test_users]
            test_users = set(test_users)

        for user_idx, df in self.trans.groupby('user'):
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
    def train_data(self):
        if not self._padded:
            self.padding()

        # Note that padding is pre using latest transactions
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            self.train_wrapper.trans_indices, maxlen=self.config.max_history_length,
            padding='pre', truncating='pre', value=-1
        ).reshape([-1])
        item_indices = self.trans.iloc[trans_indices]['item']

        profile = self.profile_data[self.train_wrapper.user_indices]
        info = self.info_data[item_indices].reshape(
            [len(self.train_wrapper), self.config.max_history_length, -1])
        context = self.context_data[trans_indices].reshape(
            [len(self.train_wrapper.user_indices), self.config.max_history_length, -1])
        desc = self.desc_data[item_indices].reshape(
            [len(self.train_wrapper), self.config.max_history_length, -1]) if self.include_desc else None
        image = self.image_path[item_indices].reshape(
            [len(self.train_wrapper), self.config.max_history_length]) if self.include_image else None

        data = {
            'info': info,
            'desc': desc,
            'image_path': image,
            'profile': profile,
            'context': context
        }
        return data

    @property
    def item_data(self):
        if not self._padded:
            self.padding()

        data = {
            'info': self.info_data,
            'desc': self.desc_data,
            'image_path': self.image_path
        }
        return data

    @property
    def desc_data(self):
        if not self.include_desc:
            return None

        assert self._processed
        token_ids = tf.keras.preprocessing.sequence.pad_sequences(
            self.items['desc'].to_list(), maxlen=self.config.max_desc_length,
            padding='post', truncating='post', dtype=np.int32, value=0
        )
        if self._padded:
            token_ids[-1] *= 0

        return token_ids

    @property
    def info_data(self):
        assert self._processed
        return np.asarray(self.items['info'].to_list(), np.int32)

    @property
    def image_path(self):
        if not self.include_image:
            return None

        return np.asarray(self.items['image'])

    @property
    def profile_data(self):
        assert self._processed
        return np.asarray(self.users['profile'].to_list(), dtype=np.int32)

    @property
    def context_data(self):
        assert self._processed
        return np.asarray(self.trans['context'].to_list(), dtype=np.int32)

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

    def padding(self):
        # pad items
        padding = {col: None for col in self.items}
        padding['info'] = (0,) * len(self.item_feature_dict)
        padding['desc'] = (0,) * self.config.max_desc_length
        padding['image'] = ''
        self.items.loc[-1] = padding

        # pad users
        padding = {col: None for col in self.users}
        padding['profile'] = (0,) * len(self.user_feature_dict)
        self.users.loc[-1] = padding

        # pad transactions
        padding = {col: None for col in self.users}
        padding['context'] = (0,) * len(self.trans_feature_dict)
        padding['item'] = -1
        padding['user'] = -1
        self.trans.loc[-1] = padding

        self._padded = True

    def _process_item_features(self, tokenizer: BertTokenizer = None):
        print('Process item features ...', end='')
        info = []
        for key, feat_map in self.item_feature_dict.items():
            info.append(self.items[key].map(feat_map))
            del self.items[key]

        self.items['info'] = list(zip(*info))
        if 'desc' in self.items and tokenizer is not None:
            # Wether or not to tokenize items' descriptions
            self.items['desc'] = tokenizer(
                self.items['desc'].to_list(),
                max_length=self.config.max_desc_length,
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']

        print('Done!')

    def _process_user_features(self):
        print('Process user features ...', end='')
        profile = []
        for key, feat_map in self.user_feature_dict.items():
            profile.append(self.users[key].map(feat_map))
            del self.users[key]

        self.users['profile'] = list(zip(*profile))
        print('Done!')

    def _process_transaction_features(self):
        print('Process transaction features ...', end='')
        context = []
        for key, feat_map in self.trans_feature_dict.items():
            context.append(self.trans[key].map(feat_map))
            del self.trans[key]

        self.trans['context'] = list(zip(*context))
        print('Done!')

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

    def save_feature_dict(self, save_dir):
        # Save feature dict to direction
        with open(os.path.join(save_dir, 'item_feature_dict.json'), 'w', encoding='utf8') as fp:
            json.dump(self.item_feature_dict, fp)
        with open(os.path.join(save_dir, 'user_feature_dict.json'), 'w', encoding='utf8') as fp:
            json.dump(self.user_feature_dict, fp)
        with open(os.path.join(save_dir, 'trans_feature_dict.json'), 'w', encoding='utf8') as fp:
            json.dump(self.trans_feature_dict, fp)

    def load_feature_dict(self, load_dir):
        # Load feature dict from direction
        with open(os.path.join(load_dir, 'item_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.item_feature_dict = json.load(fp)
        with open(os.path.join(load_dir, 'user_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.user_feature_dict = json.load(fp)
        with open(os.path.join(load_dir, 'trans_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.trans_feature_dict = json.load(fp)
        self._display_feature_info()

    def train_dataset(self, batch_size: int):
        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(
            self.train_data).shuffle(batch_size*2).map(
            self.process_train, autotune).batch(batch_size)

        return dataset

    def item_dataset(self, batch_size=32):
        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(self.item_data).map(
            self.process_infer, autotune).batch(batch_size)

        return dataset

    def read_image(self, img_path):
        img_bytes = tf.io.read_file(img_path)
        image = tf.image.decode_image(img_bytes, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.resize_image:
            image = tf.image.resize(
                image, size=(self.config.image_height, self.config.image_width)
            )

        return image

    def process_train(self, inputs):
        """ Train images are 2D image path 
        """
        def _pad_image():
            return tf.zeros((self.config.image_height, self.config.image_width, 3), dtype=tf.float32)

        def _get_image(img):
            return tf.cond(tf.not_equal(img, ''), lambda: self.read_image(img), _pad_image)

        image_path = inputs.pop('image_path', None)
        if image_path is not None:
            inputs['image'] = tf.map_fn(_get_image, image_path, dtype=tf.float32)
            inputs['image'].set_shape(
                (self.config.max_history_length, self.config.image_height, self.config.image_width, 3))

        return inputs

    def process_infer(self, inputs):
        """ Infer images are list of image path 
        """
        image_path = inputs.pop('image_path', None)
        if image_path is not None:
            inputs['image'] = self.read_image(inputs['image_path'])
            inputs['image'].set_shape((self.config.image_height, self.config.image_width, 3))

        return inputs


def build_config(config):
    if isinstance(config, dict):
        Config = namedtuple('Config', config.keys())
        config = Config(**config)

    return config
