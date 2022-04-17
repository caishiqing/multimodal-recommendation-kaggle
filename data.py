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
                 image_dir: str = None,
                 tfrecord_dir: str = None,
                 resize_image: bool = False):

        assert 'id' in items and 'id' in users
        assert 'item' in trans and 'user' in trans

        self.item_feature_dict = OrderedDict()
        self.user_feature_dict = OrderedDict()
        self.trans_feature_dict = OrderedDict()
        self.train_wrapper = DataWrapper()
        self.test_wrapper = DataWrapper()
        self._padded = False
        self._processed = False

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

        # add tfrecord path column to item
        if tfrecord_dir is not None:
            self.items['tfrecord'] = self.items['id'].apply(lambda x: os.path.join(tfrecord_dir, str(x)+'.tfrecord'))

        # add image path to item
        if image_dir is not None:
            self.items['image'] = self.items['id'].apply(lambda x: os.path.join(image_dir, str(x)+'.jpg'))

    def prepare_features(self, tokenizer: BertTokenizer = None, padding_desc=False):
        if not self._processed:
            self._process_item_features(tokenizer, padding_desc)
            self._process_user_features()
            self._process_transaction_features()
            self._processed = True
            self.padding()
        else:
            print("Features are aleady prepared.")

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
    def tfrecord_path(self):
        if 'tfrecord' not in self.items:
            return None

        return np.asarray(self.items['tfrecord'])

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

    # @property
    # def infer_wrapper(self):
    #     wrapper = DataWrapper()
    #     for i in self.users.index:
    #         wrapper.append(i, [])

    #     for user_indice, df in self.trans.groupby('user'):
    #         trans_indices = df.index.to_list()
    #         wrapper.set_value(user_indice, trans_indices)

    #     return wrapper

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

    @property
    def include_image(self):
        return 'image' in self.items or 'image_path' in self.items

    @property
    def include_desc(self):
        return 'desc' in self.items

    def padding(self):
        # pad items
        padding = {col: None for col in self.items}
        padding['id'] = 'padding'
        padding['info'] = (0,) * len(self.item_feature_dict)
        padding['desc'] = (0,) * self.config.max_desc_length
        if 'tfrecord' in self.items:
            tfrecord_dir = os.path.split(self.items['tfrecord'][0])[0]
            padding['tfrecord'] = os.path.join(tfrecord_dir, 'padding.tfrecord')
        if 'image' in self.items:
            image_dir = os.path.split(self.items['image'][0])[0]
            padding['image'] = os.path.join(tfrecord_dir, 'padding.image')
        self.items.loc[-1] = padding

        # pad users
        padding = {col: None for col in self.users}
        padding['id'] = 'padding'
        padding['profile'] = (0,) * len(self.user_feature_dict)
        self.users.loc[-1] = padding

        # pad transactions
        padding = {col: None for col in self.users}
        padding['context'] = (0,) * len(self.trans_feature_dict)
        padding['item'] = -1
        padding['user'] = -1
        self.trans.loc[-1] = padding

        self._padded = True

    def _process_item_features(self,
                               tokenizer: BertTokenizer = None,
                               padding_desc=False):

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
                truncation=True,
                padding=padding_desc,
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
            self.item_feature_dict = json.load(fp)
        with open(os.path.join(load_dir, 'user_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.user_feature_dict = json.load(fp)
        with open(os.path.join(load_dir, 'trans_feature_dict.json'), 'r', encoding='utf8') as fp:
            self.trans_feature_dict = json.load(fp)
        self._display_feature_info()

    def train_dataset(self, batch_size: int = 8):
        assert self._processed and self._padded
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            self.train_wrapper.trans_indices, maxlen=self.config.max_history_length,
            padding='pre', truncating='pre', value=-1
        ).reshape([-1])
        item_indices = self.trans.iloc[trans_indices]['item']

        profile = tf.data.Dataset.from_tensor_slices(
            self.profile_data[self.train_wrapper.user_indices]).unbatch()
        context = tf.data.Dataset.from_tensor_slices(
            self.context_data[trans_indices]).unbatch().batch(self.config.max_history_length)

        tfrecord_path = self.tfrecord_path
        image_path = self.image_path
        assert tfrecord_path is not None or image_path is not None

        autotune = tf.data.experimental.AUTOTUNE
        if tfrecord_path is not None:
            print('Building dataset with items from TFRecords ...', end='')
            item_record_paths = tfrecord_path[item_indices]
            item = tf.data.TFRecordDataset(
                item_record_paths, num_parallel_reads=autotune
            ).map(self._parse_item_tfrecord, autotune).unbatch().batch(self.config.max_history_length)

            dataset = tf.data.Dataset.zip((profile, context, item)).map(
                self._process_tfrecord).shuffle(2*batch_size).batch(batch_size)
            print('Done.')
        else:
            print('Building dataset with items from image files ...', end='')
            item_image_paths = image_path[item_indices]
            info = tf.data.Dataset.from_tensor_slices(
                self.info_data[item_indices]).unbatch().batch(self.config.max_history_length)
            desc = tf.data.Dataset.from_tensor_slices(
                self.desc_data[item_indices]).unbatch().batch(self.config.max_history_length)
            image = tf.data.Dataset.from_tensor_slices(
                item_image_paths).map(self._read_image, autotune).unbatch().batch(self.config.max_history_length)
            dataset = tf.data.Dataset.zip((profile, context, info, desc, image)).map(
                self._process_train).shuffle(2*batch_size).batch(batch_size)
            print('Done!')

        return dataset

    def item_dataset(self, batch_size: int = 32):
        assert self._processed and self._padded
        autotune = tf.data.experimental.AUTOTUNE
        tfrecord_path = self.tfrecord_path
        image_path = self.image_path
        assert tfrecord_path is not None or image_path is not None

        if tfrecord_path is not None:
            print('Building item dataset from TFRecords ...', end='')
            dataset = tf.data.TFRecordDataset(
                tfrecord_path, num_parallel_reads=autotune
            ).map(self._parse_item_tfrecord, autotune).batch(batch_size)
            print('Done.')
        else:
            print('Building item dataset from image files ...', end='')
            info = tf.data.Dataset.from_tensor_slices(self.info_data)
            desc = tf.data.Dataset.from_tensor_slices(self.desc_data)
            image = tf.data.Dataset.from_tensor_slices(image_path).map(self._read_image, autotune)
            dataset = tf.data.Dataset.zip((info, desc, image)).map(self._process_item).batch(batch_size)
            print('Done.')

        return dataset

    def _decode_image(self, img_bytes):
        image = tf.image.decode_image(img_bytes, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    def _read_image(self, img_path):
        img_bytes = tf.io.read_file(img_path)
        image = self._decode_image(img_bytes)
        if self.resize_image:
            image = tf.image.resize(
                image, size=(self.config.image_height, self.config.image_width)
            )

        return image

    def _process_train(self, *inputs):
        """ Train images are 2D image path 
        """
        profile, context, info, desc, image = inputs
        inputs = {
            'profile': profile,
            'context': context,
            'info': info,
            'desc': desc,
            'image': image
        }
        return inputs

    def _process_tfrecord(self, *inputs):
        profile, context, item = inputs
        inputs = {
            'profile': profile,
            'context': context,
            'image': item['image'],
            'desc': item['desc'],
            'info': item['info']
        }
        return inputs

    def _process_item(self, *inputs):
        info, desc, image = inputs
        return {
            'info': info,
            'desc': desc,
            'image': image
        }

    def _process_infer(self, inputs):
        """ Infer images are list of image path 
        """
        image_path = inputs.pop('image_path', None)
        if image_path is not None:
            inputs['image'] = self._read_image(inputs['image_path'])
            inputs['image'].set_shape((self.config.image_height, self.config.image_width, 3))

        return inputs

    def _parse_item_tfrecord(self, item_tfrecord_proto):
        example = tf.io.parse_single_example(item_tfrecord_proto, self.item_schema)
        example['image'] = self._decode_image(example['image'])
        return example

    @property
    def item_schema(self):
        schema = {
            'info': tf.io.FixedLenFeature([len(self.info_size)], tf.int64),
            'desc': tf.io.FixedLenFeature([self.config.max_desc_length], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        return schema


def build_config(config):
    if isinstance(config, dict):
        Config = namedtuple('Config', config.keys())
        config = Config(**config)

    return config
