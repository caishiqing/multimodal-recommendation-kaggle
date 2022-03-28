from collections import OrderedDict
from os import O_EXCL
import tensorflow as tf
import numpy as np
import random


class DataLoader(object):
    def __init__(self,
                 config,
                 items,
                 users,
                 transactions,
                 user_feature_dict,
                 item_feature_dict,
                 tokenizer):

        self.config = config
        self.items = items
        self.users = users
        self.item_id2index = OrderedDict([(item['id'], i) for i, item in enumerate(items)])
        self.user_id2index = OrderedDict([(user['id'], i) for i, user in enumerate(users)])
        self.transactions = transactions
        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.tokenizer = tokenizer

    def train_dataset(self, batch_size=32):
        info, desc, image, profile = [], [], [], []
        random.shuffle(self.transactions)
        max_image_path_length = max([len(item['image']) for item in self.items])
        for trans in self.transactions:
            user_id = trans['user_id']
            item_ids = trans['items']
            user = self.users[self.user_id2index[user_id]]

            _profile = []
            for key, feat_map in self.user_feature_dict.items():
                val = user.get(key)
                feat_id = feat_map.get(val, 0)
                _profile.append(feat_id)
            profile.append(_profile)

            _info, _desc, _image = [], [], []
            for item_id in item_ids:
                item = self.items[self.item_id2index[item_id]]
                _info.append(self._attribute_feat(item, self.item_feature_dict))

                desc_token_ids = self.tokenizer(
                    item['desc'],
                    truncation=True,
                    max_length=self.config.max_desc_length,
                    padding='max_length'
                )['input_ids']
                _desc.append(desc_token_ids)

                image_path = item['image']
                _image.append(image_path)

            info.append(_info)
            desc.append(_desc)
            image.append(_image)

        info = tf.keras.preprocessing.sequence.pad_sequences(
            info, maxlen=self.config.max_history_length,
            padding='post', truncating='post'
        )
        desc = tf.keras.preprocessing.sequence.pad_sequences(
            desc, maxlen=self.config.max_history_length,
            padding='post', truncating='post'
        )
        image = tf.keras.preprocessing.sequence.pad_sequences(
            image, maxlen=self.config.max_history_length,
            padding='post', truncating='post',
            dtype=f'<U{max_image_path_length}', value=''
        )
        profile = np.asarray(profile, dtype=np.int32)

        autotune = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                'info': info,
                'desc': desc,
                'img_path': image,
                'profile': profile,
            }
        ).map(self.process_train, autotune).batch(batch_size)

        return dataset

    def infer_dataset(self, batch_size=32):
        info, desc, image, profile = [], [], [], []
        for item in self.items:
            info.append(self._attribute_feat(item, self.item_feature_dict))
            desc_token_ids = self.tokenizer(
                item['desc'],
                truncation=True,
                max_length=self.config.max_desc_length,
                padding='max_length'
            )['input_ids']
            desc.append(desc_token_ids)
            image.append(item['image'])

        for user in self.users:
            profile.append(self._attribute_feat(user, self.user_feature_dict))

        autotune = tf.data.experimental.AUTOTUNE
        item_dataset = tf.data.Dataset.from_tensor_slices(
            {
                'info': np.asarray(info, np.int32),
                'desc': np.asarray(desc, np.int32),
                'image_path': np.asarray(image)
            }
        ).map(self.process_infer, autotune).batch(batch_size)
        user_dataset = np.asarray(profile, np.int32)

        return item_dataset, user_dataset

    def _attribute_feat(self, item, feature_dict):
        attr = []
        for key, feat_map in feature_dict.items():
            val = item[key]
            feat_id = feat_map.get(val, 0)
            attr.append(feat_id)

        return attr

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

        inputs['image'] = tf.map_fn(_get_image, inputs['image_path'], dtype=tf.float32)
        return inputs

    def process_infer(self, inputs):
        """ Infer images are list of image path 
        """
        inputs['image'] = self.read_image(inputs['image_path'])
        return inputs
