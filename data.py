from transformers import BertTokenizer, BertConfig
from collections import OrderedDict, namedtuple
from typing import List, Union
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json
import os


class RecData(object):
    item_feature_dict = OrderedDict()
    user_feature_dict = OrderedDict()
    trans_feature_dict = OrderedDict()
    train = list()
    test = list()
    _padded = False
    _processed = False

    def __init__(self,
                 items: pd.DataFrame,
                 users: pd.DataFrame,
                 trans: pd.DataFrame,
                 config: Union[dict, BertConfig],
                 feature_path: str = None):

        assert 'id' in items and 'id' in users
        assert 'item' in trans and 'user' in trans
        self.config = build_config(config)

        self.items = items.reset_index(drop=True)
        self.users = users.reset_index(drop=True)
        self.trans = trans

        # items and users use reseted index
        self.item_index_map = OrderedDict([(id, i) for i, id in enumerate(self.items['id'])])
        self.user_index_map = OrderedDict([(id, i) for i, id in enumerate(self.users['id'])])
        self.trans.reset_index(drop=True, inplace=True)
        self.trans['item'] = self.trans['item'].map(self.item_index_map)
        self.trans['user'] = self.trans['user'].map(self.user_index_map)

        # process features
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

    def process_features(self, tokenizer: BertTokenizer = None):
        if not self._processed:
            self._process_item_features(tokenizer)
            self._process_user_features()
            self._process_transaction_features()
            self._processed = True

        self._display_feature_info()

    def prepare(self, mode: str, test_users: list):
        # Process and get train/test data from transactions
        self.train, self.test = [], []
        test_users = [self.user_index_map[user] for user in test_users]
        self.users['split'] = 'train'
        self.users['split'][test_users] = 'test'

        with tqdm(total=len(self.trans['user'].unique()), desc='Process transactions') as pbar:
            for user, df in self.trans.groupby('user', sort=False):
                pbar.update(1)

                split = self.users['split'][user]
                indices = df.index.to_list()

                if split == 'test':
                    ground_truth = []
                    if mode == 'train' and len(indices) > self.config.predict_length:
                        # Cut last transaction as target if mode is train
                        pred_ids = indices[-self.config.predict_length:]
                        ground_truth = df['item'][pred_ids].to_list()
                        indices = indices[:len(indices)-self.config.predict_length]

                    if mode == 'test' or ground_truth:
                        self.test.append(
                            {
                                'user': user,
                                'indices': indices[-self.config.max_history_length:],
                                'ground_truth': ground_truth
                            }
                        )

                if mode == 'train' and len(indices) < max(
                        self.config.min_train_length, self.config.max_history_length):
                    continue

                if mode == 'train':
                    i, j = 0, 0
                    while j < len(indices)-1:
                        if len(indices) - i < 2 * self.config.max_history_length:
                            # assert no cross history
                            i = len(indices) - self.config.max_history_length - 1

                        j = i + self.config.max_history_length
                        while i > 0 and j >= len(indices):
                            i -= 1
                            j -= 1
                        self.train.append({'user': user, 'indices': indices[i:j]})
                        i = j

        print('Train samples: {}'.format(len(self.train)))
        print('Test samples: {}'.format(len(self.test)))

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
            self.items['desc'] = self.items['desc'].apply(
                lambda x: tokenizer(
                    x, truncation=True, padding='max_length',
                    max_length=self.config.max_desc_length
                )['input_ids']
            )
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
            if col in ('id', 'image', 'image_path', 'desc'):
                continue
            vals = set(self.items[col])
            self.item_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(sorted(vals))])

        for col in self.users.columns:
            if col in ('split', 'id'):
                continue
            vals = set(self.users[col])
            self.user_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(sorted(vals))])

        for col in self.trans.columns:
            if col in ('user', 'item'):
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


class DataLoader(object):
    def __init__(self, data: RecData):

        self.config = data.config
        self.data = data

    def train_dataset(self, batch_size=32):
        if not self.data._padded:
            self.data.padding()

        random.shuffle(self.data.train)
        user_indices, trans_indices = [], []
        for record in self.data.train:
            user_indices.append(record['user'])
            trans_indices.append(record['indices'])

        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            trans_indices, maxlen=self.config.max_history_length,
            padding='post', truncating='post', value=-1
        ).reshape(len(self.data.train) * self.config.max_history_length)
        item_indices = self.data.trans.iloc[trans_indices]['item']

        profile = np.asarray(self.data.users.iloc[user_indices]['profile'].to_list(), dtype=np.int32)
        info = np.asarray(self.data.items.iloc[item_indices]['info'].to_list(), dtype=np.int32).reshape(
            [len(self.data.train), self.config.max_history_length, -1])
        context = np.asarray(self.data.trans.iloc[trans_indices]['context'].to_list(), dtype=np.int32).reshape(
            [len(self.data.train), self.config.max_history_length, -1])
        desc = np.asarray(self.data.items.iloc[item_indices]['desc'].to_list(), dtype=np.int32).reshape(
            [len(self.data.train), self.config.max_history_length, -1]) if self.data.include_desc else None
        image = np.asarray(self.data.items.iloc[item_indices]['image'].to_list(), dtype=np.unicode_).reshape(
            [len(self.data.train), self.config.max_history_length]) if self.data.include_image else None

        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                'info': info,
                'desc': desc,
                'image_path': image,
                'profile': profile,
                'context': context
            }
        ).map(self.process_train, autotune).batch(batch_size)
        return dataset

    def item_dataset(self, batch_size=32):
        if not self.data._padded:
            self.data.padding()

        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                'info': np.asarray(self.data.items['info'], np.int32),
                'desc': np.asarray(self.data.items['desc'], np.int32) if self.data.include_desc else None,
                'image_path': np.asarray(self.data.items['image']) if self.data.include_image else None
            }
        ).map(self.process_infer, autotune).batch(batch_size)

        return dataset

    def infer_dataset(self, item_vectors, batch_size=512):
        user_indices, trans_indices, ground_truths = [], [], []
        for record in self.data.test:
            user_indices.append(record['user'])
            trans_indices.append(record['indices'])
            ground_truths.append(record['groud_truth'])

        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            trans_indices, maxlen=self.config.max_history_length,
            padding='post', truncating='post', value=-1
        ).reshape(len(self.data.train) * self.config.max_history_length)
        item_indices = self.data.trans.iloc[trans_indices]['item']

        profile = np.asarray(self.data.users.iloc[user_indices]['profile'].to_list(), dtype=np.int32)
        context = np.asarray(self.data.trans.iloc[trans_indices]['context'].to_list(), dtype=np.int32).reshape(
            [len(self.data.train), self.config.max_history_length, -1])

        dataset = {
            'profile': profile,
            'context': context,
            'item_indices': item_indices,
            'ground_truths': ground_truths
        }
        return dataset

    def read_image(self, img_path):
        img_bytes = tf.io.read_file(img_path)
        image = tf.image.decode_image(img_bytes, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(
            image, size=(self.config.image_height, self.config.image_width))

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

        return inputs

    def process_infer(self, inputs):
        """ Infer images are list of image path 
        """
        image_path = inputs.pop('image_path', None)
        if image_path is not None:
            inputs['image'] = self.read_image(inputs['image_path'])

        return inputs


def build_config(config):
    if isinstance(config, dict):
        Config = namedtuple('Config', config.keys())
        config = Config(**config)

    return config
