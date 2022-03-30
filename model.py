import numpy as np
from transformers import TFBertModel, BertConfig
from tensorflow.keras import layers
from criterion import CircleLoss
import tensorflow as tf
import os


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


class AttributeEmbedding(layers.Layer):
    """ 属性嵌入层 """

    def __init__(self, group_size, embed_dim=512, **kwargs):
        super(AttributeEmbedding, self).__init__(**kwargs)
        self.group_size = group_size
        self.embed_dim = embed_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name='{}_embedding'.format(self.name),
            shape=(sum(self.group_size), self.embed_dim),
            initializer='uniform',
            dtype=tf.float32
        )
        _cum = [0] + self.group_size[:-1]
        self.cum = tf.cast(tf.cumsum(_cum), tf.int32)[tf.newaxis, :]
        super(AttributeEmbedding, self).build(input_shape)

    def call(self, inputs):
        indices = inputs + self.cum
        embeds = tf.gather(self.embedding, indices)
        return tf.reduce_sum(embeds, axis=-2)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim

    def get_config(self):
        config = super(AttributeEmbedding, self).get_config()
        config['group_size'] = self.group_size
        config['embed_dim'] = self.embed_dim
        return config


class Item(tf.keras.Model):
    """ 商品模型 """

    def __init__(self, attribute_size, embed_dim=512, **kwargs):
        super(Item, self).__init__(**kwargs)
        self.image_model = Image(embed_dim, kwargs.pop('image_weights', None))
        self.desc_model = Desc(embed_dim, kwargs.pop('bert_path', None))
        self.info_model = AttributeEmbedding(attribute_size, embed_dim)
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


class Items(Item):
    """ 商品序列模型 """

    def __init__(self, attribute_size, embed_dim=512, **kwargs):
        super(Items, self).__init__(attribute_size, embed_dim=512, **kwargs)
        self.image_model = layers.TimeDistributed(self.image_model)
        self.desc_model = layers.TimeDistributed(self.desc_model)
        self.info_model = layers.TimeDistributed(self.info_model)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return tf.reduce_any(tf.not_equal(inputs['desc'], 0), axis=-1)

    def call(self, inputs, training=None):
        return super(Items, self).call(inputs, training=training)


class User(tf.keras.Model):
    """ 用户模型 """

    def __init__(self, profile_size, embed_dim=512, **kwargs):
        dropout = kwargs.pop('dropout', 0.0)
        recurrent_dropout = kwargs.pop('recurrent_dropout', dropout)
        super(User, self).__init__(**kwargs)
        self.profile_model = tf.keras.Sequential([AttributeEmbedding(profile_size, embed_dim)])
        self.history_model = layers.GRU(
            embed_dim, recurrent_activation='hard_sigmoid',
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=True, return_state=True
        )

    def call(self, inputs, training=None):
        initial_state = inputs.get('initial_state')
        if initial_state is None:
            initial_state = self.profile_model(inputs['profile'])

        y, h = self.history_model(
            inputs['items'],
            initial_state=initial_state,
            training=training
        )
        return y, h

    def infer_initial_state(self, profile, batch_size=32):
        return self.profile_model.predict(profile, batch_size=batch_size)


def build_train_model(
    attribute_size, profile_size,
    embed_dim=512, max_desc_length=8,
    max_history_length=64,
    image_shape=(224, 224),
    **kwargs
):
    inputs = {
        'info': layers.Input(shape=(max_history_length, len(attribute_size)), dtype=tf.int32),
        'desc': layers.Input(shape=(max_history_length, max_desc_length), dtype=tf.int32),
        'image': layers.Input(shape=(max_history_length,) + tuple(image_shape) + (3,), dtype=tf.float32),
        'profile': layers.Input(shape=(len(profile_size),), dtype=tf.int32)
    }
    items = Items(attribute_size, embed_dim, name='Items')(inputs)
    state_seq, _ = User(profile_size, embed_dim, name='User')(
        {'profile': inputs['profile'], 'items': items})

    batch_size = tf.shape(inputs['desc'])[0]
    batch_idx = tf.range(0, batch_size)
    length_idx = tf.range(0, max_history_length)
    a = batch_idx[:, tf.newaxis, tf.newaxis, tf.newaxis]
    b = length_idx[tf.newaxis, :, tf.newaxis, tf.newaxis]
    c = batch_idx[tf.newaxis, tf.newaxis, :, tf.newaxis]
    d = length_idx[tf.newaxis, tf.newaxis, tf.newaxis, :]

    prd_mask = tf.logical_and(tf.equal(a, c), tf.greater_equal(b, d))
    prd_mask = tf.reshape(prd_mask, [batch_size, max_history_length, -1])
    prd_mask = tf.logical_not(prd_mask)  # mask history item
    pad_mask = tf.reduce_any(tf.not_equal(inputs['desc'], 0), axis=-1)
    pad_mask = tf.reshape(pad_mask, [-1])[tf.newaxis, tf.newaxis, :]
    mask = tf.logical_and(pad_mask, prd_mask)  # (batch, len, batch * len)

    # compute logits
    item = tf.nn.l2_normalize(items, axis=-1)  # (batch, len, dim)
    user = tf.nn.l2_normalize(state_seq, axis=-1)  # (batch, len, dim)
    item = tf.reshape(item, [-1, embed_dim])  # (batch * len, dim)
    logits = tf.matmul(user, item, transpose_b=True)  # (batch, len, batch * len)
    logits = tf.boolean_mask(logits, mask)

    # compute labels
    labels = tf.tile(tf.equal(a, c), [1, max_history_length, 1, max_history_length])
    labels = tf.cast(tf.reshape(labels, [batch_size, max_history_length, -1]), tf.float32)
    labels = tf.boolean_mask(labels, mask)

    loss = CircleLoss(
        margin=kwargs.get('margin', 0.25),
        gamma=kwargs.get('gamma', 32)
    )(labels, (1 + logits) / 2)
    model = tf.keras.Model(inputs=inputs, outputs=state_seq)
    model.add_loss(loss)
    return model


inputs = {
    'image': layers.Input(shape=(10, 223, 223, 3), dtype=tf.float32),
    'desc': layers.Input(shape=(10, 6), dtype=tf.int32),
    'info': layers.Input(shape=(10, 4), dtype=tf.int32)
}

user_inputs = {
    'profile': layers.Input(shape=(4,), dtype=tf.int32),
    'items': layers.Input(shape=(10, 512))
}
user_model = User([4, 4, 4, 4], 512)
y, h = user_model(user_inputs)
user_model.summary()

user_inputs = {
    'initial_state': np.random.random((1, 512)),
    'items': np.random.random((1, 10, 512))
}
y, h = user_model.predict(user_inputs)
print(y.shape)
