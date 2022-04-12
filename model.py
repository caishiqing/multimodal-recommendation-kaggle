from sqlalchemy import over
from transformers import TFBertModel, BertConfig
from tensorflow.keras import layers
from evaluate import CircleLoss
import tensorflow as tf
import os


class AttributeEmbedding(layers.Layer):
    """ 属性嵌入层 """

    def __init__(self, size, embed_dim=512, **kwargs):
        super(AttributeEmbedding, self).__init__(**kwargs)
        self.size = size
        self.embed_dim = embed_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name='{}_embedding'.format(self.name),
            shape=(sum(self.size), self.embed_dim),
            initializer='normal',
            dtype=tf.float32
        )
        _cum = [0] + self.size[:-1]
        self.cum = tf.cast(tf.cumsum(_cum), tf.int32)[tf.newaxis, :]
        super(AttributeEmbedding, self).build(input_shape)

    def call(self, inputs):
        indices = inputs + self.cum
        embeds = tf.gather(self.embedding, indices)
        return tf.reduce_mean(embeds, axis=-2)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim

    def get_config(self):
        config = super(AttributeEmbedding, self).get_config()
        config['size'] = self.size
        config['embed_dim'] = self.embed_dim
        return config


class Image(layers.Layer):
    """ 商品图片模型 """

    def __init__(self, embed_dim=512, image_weights=None, **kwargs):
        super(Image, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False, pooling='max',  weights=image_weights)
        self.ln = layers.LayerNormalization()
        self.dense = layers.Dense(self.embed_dim)

    def call(self, inputs, training=None):
        x = self.backbone(inputs, training=training)
        img = self.ln(x)
        return self.dense(img)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim

    def get_config(self):
        config = super(Image, self).get_config()
        config['embed_dim'] = self.embed_dim
        return config


class Desc(layers.Layer):
    """ 商品描述模型 """

    def __init__(self, embed_dim=512, bert_path=None, **kwargs):
        super(Desc, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        bert_path = bert_path or 'bert-base-uncased'
        if os.path.isdir(bert_path):
            config = BertConfig.from_json_file(os.path.join(bert_path, 'config.json'))
            self.backbone = TFBertModel(config)
        else:
            self.backbone = TFBertModel.from_pretrained(bert_path)
        self.dense = layers.Dense(self.embed_dim)
        self.supports_masking = True

    def call(self, inputs, training=None):
        attention_mask = tf.not_equal(inputs, 0)
        x = self.backbone(
            input_ids=inputs,
            attention_mask=attention_mask,
            training=training
        ).last_hidden_state[:, 0, :]
        x = self.dense(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim

    def get_config(self):
        config = super(Desc, self).get_config()
        config['embed_dim'] = self.embed_dim
        return config


class Item(tf.keras.Model):
    """ 商品模型 """

    def __init__(self, info_size, embed_dim=512, **kwargs):
        image_weights = kwargs.pop('image_weights', None)
        bert_path = kwargs.pop('bert_path', None)
        super(Item, self).__init__(**kwargs)
        self.image_model = Image(embed_dim, image_weights=image_weights, name='Image')
        self.desc_model = Desc(embed_dim, bert_path=bert_path, name='Desc')
        self.info_model = AttributeEmbedding(info_size, embed_dim, name='Info')
        self.ln = layers.LayerNormalization()
        self.dense = layers.Dense(embed_dim)

    def call(self, inputs, training=None):
        img_embed = self.image_model(inputs['image'], training=training)
        desc_embed = self.desc_model(inputs['desc'], training=training)
        info_embed = self.info_model(inputs['info'])
        x = layers.Add()([img_embed, desc_embed, info_embed])
        x = self.ln(x)
        x = self.dense(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim


class Items(tf.keras.Model):
    """ 商品序列模型 """

    def __init__(self, item_model, **kwargs):
        super(Items, self).__init__(**kwargs)
        self.item_model = item_model
        self.image_model = layers.TimeDistributed(item_model.image_model)
        self.desc_model = layers.TimeDistributed(item_model.desc_model)
        self.info_model = layers.TimeDistributed(item_model.info_model)
        self.ln = layers.TimeDistributed(item_model.ln)
        self.dense = layers.TimeDistributed(item_model.dense)
        self.supports_masking = True

    def call(self, inputs, training=None, mask=None):
        img_embed = self.image_model(inputs['image'], training=training)
        desc_embed = self.desc_model(inputs['desc'], training=training)
        info_embed = self.info_model(inputs['info'])
        x = layers.Add()([img_embed, desc_embed, info_embed])
        x = self.ln(x)
        x = self.dense(x)

        if mask is not None:
            m = tf.expand_dims(tf.cast(mask, x.dtype), -1)
            x *= m

        return x

    def compute_mask(self, inputs, mask=None):
        return tf.reduce_any(tf.not_equal(inputs['desc'], 0), axis=-1)


class User(tf.keras.Model):
    """ 用户模型 """

    def __init__(self, profile_size, context_size, embed_dim=512, **kwargs):
        dropout = kwargs.pop('dropout', 0.0)
        recurrent_dropout = kwargs.pop('recurrent_dropout', dropout)
        super(User, self).__init__(**kwargs)
        self.profile_model = tf.keras.Sequential([AttributeEmbedding(profile_size, embed_dim)], name='Profile')
        self.trans_model = layers.TimeDistributed(AttributeEmbedding(context_size, embed_dim), name='Context')
        self.masking = layers.Masking(0.0)
        self.history_model = layers.GRU(
            embed_dim, recurrent_activation='hard_sigmoid',
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=True, return_state=True
        )

    def call(self, inputs, training=None):
        initial_state = inputs.get('initial_state')
        if initial_state is None:
            initial_state = self.profile_model(inputs['profile'])

        context = self.trans_model(inputs['context'])
        items = self.masking(inputs['items'])
        x = layers.Add()([context, items])
        y, h = self.history_model(x, initial_state=initial_state, training=training)

        return y, h

    def infer_initial_state(self, profile, batch_size=32):
        return self.profile_model.predict(profile, batch_size=batch_size)


# def build_train_model(
#     info_size, profile_size, context_size,
#     embed_dim=512, max_desc_length=8,
#     max_history_length=64, predict_length=12,
#     image_height=224, image_width=224,
#     **kwargs
# ):
#     # sequence of items and transactions context of user history
#     inputs = {
#         'info': layers.Input(shape=(max_history_length, len(info_size)), dtype=tf.int32),
#         'desc': layers.Input(shape=(max_history_length, max_desc_length), dtype=tf.int32),
#         'image': layers.Input(shape=(max_history_length, image_height, image_width, 3), dtype=tf.float32),
#         'profile': layers.Input(shape=(len(profile_size),), dtype=tf.int32),
#         'context': layers.Input(shape=(max_history_length, len(context_size)), dtype=tf.int32)
#     }
#     item_model = Item(
#         info_size, embed_dim, name='Item',
#         bert_path=kwargs.pop('bert_path', None),
#         image_weights=kwargs.pop('image_weights', None)
#     )
#     user_model = User(
#         profile_size, context_size,
#         embed_dim, name='User', **kwargs
#     )
#     items_model = Items(item_model, name='Items')
#     # call once to build item model
#     item_model(
#         {
#             'info': inputs['info'][:, 0],
#             'desc': inputs['desc'][:, 0],
#             'image': inputs['image'][:, 0]
#         }
#     )

#     item_vectors = items_model(inputs)
#     state_seq, _ = user_model(
#         {
#             'profile': inputs['profile'],
#             'items': item_vectors,
#             'context': inputs['context']
#         }
#     )

#     batch_size = tf.shape(inputs['desc'])[0]
#     batch_idx = tf.range(0, batch_size)
#     length_idx = tf.range(0, max_history_length)
#     a = batch_idx[:, tf.newaxis, tf.newaxis, tf.newaxis]
#     b = length_idx[tf.newaxis, :, tf.newaxis, tf.newaxis]
#     c = batch_idx[tf.newaxis, tf.newaxis, :, tf.newaxis]
#     d = length_idx[tf.newaxis, tf.newaxis, tf.newaxis, :]

#     prd_mask = tf.logical_and(
#         tf.equal(a, c),
#         tf.logical_or(
#             tf.greater_equal(b, d), tf.greater(d-b, predict_length))
#     )
#     prd_mask = tf.reshape(prd_mask, [batch_size, max_history_length, -1])
#     prd_mask = tf.logical_not(prd_mask)  # mask history item
#     pad_mask = tf.reduce_any(tf.not_equal(inputs['desc'], 0), axis=-1)
#     pad_mask = tf.reshape(pad_mask, [-1])[tf.newaxis, tf.newaxis, :]
#     mask = tf.logical_and(pad_mask, prd_mask)  # (batch, len, batch * len)

#     # compute logits
#     item = tf.nn.l2_normalize(item_vectors, axis=-1)  # (batch, len, dim)
#     user = tf.nn.l2_normalize(state_seq, axis=-1)  # (batch, len, dim)
#     item = tf.reshape(item, [-1, embed_dim])  # (batch * len, dim)
#     logits = tf.matmul(user, item, transpose_b=True)  # (batch, len, batch * len)
#     logits = tf.boolean_mask(logits, mask)

#     # compute labels
#     labels = tf.tile(tf.equal(a, c), [1, max_history_length, 1, max_history_length])
#     labels = tf.cast(tf.reshape(labels, [batch_size, max_history_length, -1]), tf.float32)
#     labels = tf.boolean_mask(labels, mask)

#     loss = CircleLoss(
#         margin=kwargs.get('margin', 0.25),
#         gamma=kwargs.get('gamma', 32)
#     )(labels, (1 + logits) / 2)

#     train_model = tf.keras.Model(inputs=inputs, outputs=state_seq)
#     train_model.add_loss(loss)
#     return train_model, item_model, user_model


def build_model(config):
    item_model = Item(
        config['info_size'], embed_dim=config.get('embed_dim', 256),
        bert_path=config.pop('bert_path', None),
        image_weights=config.pop('image_weights', None),
        name='Item'
    )
    user_model = User(
        config['profile_size'], config['context_size'],
        embed_dim=config.get('embed_dim', 256),
        name='User'
    )

    h, w = config.get('image_height', 224), config.get('image_width', 224)
    item_dummy_inputs = {
        'image': layers.Input(shape=(h, w, 3), dtype=tf.float32),
        'desc': layers.Input(shape=(config['max_desc_length'],), dtype=tf.int32),
        'info': layers.Input(shape=(len(config['info_size']),), dtype=tf.int32)
    }
    item_model(item_dummy_inputs)

    train_dummy_inputs = {
        'items': layers.Input(shape=(1, config.get('embed_dim', 256)), dtype=tf.float32),
        'desc': layers.Input(shape=(1, config['max_desc_length']), dtype=tf.int32),
        'info': layers.Input(shape=(1, len(config['info_size'])), dtype=tf.int32),
        'profile': layers.Input(shape=(len(config['profile_size']),), dtype=tf.int32),
        'context': layers.Input(shape=(1, len(config['context_size'])), dtype=tf.int32)
    }
    user_model(train_dummy_inputs)

    return item_model, user_model


class RecModel(tf.keras.Model):
    """ Recommendation Model for Training """

    def __init__(self, config, item_model, user_model, **kwargs):
        super(RecModel, self).__init__(**kwargs)
        self.config = config
        self.item_model = item_model
        self.items_model = Items(self.item_model, name='Items')
        self.user_model = user_model

    def compile(self, optimizer, margin=0.25, gamma=32):
        super(RecModel, self).compile(optimizer=optimizer)
        self.loss_fn = CircleLoss(margin=margin, gamma=gamma)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            item_vectors = self.items_model(inputs, training=True)
            state_seq, _ = self.user_model(
                {
                    'profile': inputs['profile'],
                    'items': item_vectors,
                    'context': inputs['context']
                },
                training=True
            )

            batch_size = tf.shape(inputs['info'])[0]
            seq_length = tf.shape(inputs['info'])[1]
            batch_idx = tf.range(0, batch_size)
            length_idx = tf.range(0, seq_length)
            a = batch_idx[:, tf.newaxis, tf.newaxis, tf.newaxis]
            b = length_idx[tf.newaxis, :, tf.newaxis, tf.newaxis]
            c = batch_idx[tf.newaxis, tf.newaxis, :, tf.newaxis]
            d = length_idx[tf.newaxis, tf.newaxis, tf.newaxis, :]

            prd_mask = tf.logical_and(
                tf.equal(a, c),
                tf.logical_or(
                    tf.greater_equal(b, d), tf.greater(d-b, self.config['predict_length']))
            )
            prd_mask = tf.reshape(prd_mask, [batch_size, seq_length, -1])
            prd_mask = tf.logical_not(prd_mask)  # mask history item
            pad_mask = tf.reduce_any(tf.not_equal(inputs['desc'], 0), axis=-1)
            pad_mask = tf.reshape(pad_mask, [-1])[tf.newaxis, tf.newaxis, :]
            mask = tf.logical_and(pad_mask, prd_mask)  # (batch, len, batch * len)

            # compute logits
            item = tf.nn.l2_normalize(item_vectors, axis=-1)  # (batch, len, dim)
            user = tf.nn.l2_normalize(state_seq, axis=-1)  # (batch, len, dim)
            item = tf.reshape(item, [-1, self.config['embed_dim']])  # (batch * len, dim)
            logits = tf.matmul(user, item, transpose_b=True)  # (batch, len, batch * len)
            logits = tf.boolean_mask(logits, mask)

            # compute labels
            labels = tf.tile(tf.equal(a, c), [1, seq_length, 1, seq_length])
            labels = tf.cast(tf.reshape(labels, [batch_size, seq_length, -1]), tf.float32)
            labels = tf.boolean_mask(labels, mask)

            loss = self.loss_fn(labels, (1 + logits) / 2)

        variables = self.item_model.trainable_weights+self.user_model.trainable_weights
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'loss': loss}

    def save_weights(self, filepath, **kwargs):
        self.item_model.save_weights(os.path.join(filepath, 'item.h5'), **kwargs)
        self.user_model.save_weights(os.path.join(filepath, 'user.h5'), **kwargs)


if __name__ == '__main__':
    config = {
        'max_history_length': 32,
        'predict_length': 12,
        'max_desc_length': 8,
        'info_size': [4, 5, 6],
        'profile_size': [2, 3],
        'context_size': [3, 5],
        'embed_dim': 64,
        'bert_path': 'bert-base-uncased',
        'image_weights': 'imagenet',
        'image_height': 128,
        'image_width': 128
    }

    item_model, user_model = build_model(config)
    item_model.summary()
    user_model.summary()
    rec_model = RecModel(config, item_model, user_model)
    rec_model.compile('adam')

    inputs = {
        'info': tf.constant([[[1, 2, 3], [2, 3, 4]]], dtype=tf.int32),
        'desc': tf.constant([[[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]]], dtype=tf.int32),
        'image': tf.random.uniform((1, 2, 128, 128, 3)),
        'profile': tf.constant([[1, 2]], dtype=tf.int32),
        'context': tf.constant([[[1, 2], [3, 4]]], dtype=tf.int32)
    }
    loss = rec_model.train_step(inputs)
