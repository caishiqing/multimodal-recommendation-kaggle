from transformers import TFBertModel, BertConfig
from tensorflow.keras import layers
from evaluate import UnifiedLoss
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
        self.dense = layers.Dense(self.embed_dim)

        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def _preprocess(self, img):
        x = tf.cast(img, tf.float32) / 256
        return (x-self.mean)/self.std

    def call(self, img, training=None):
        x = self._preprocess(img)
        x = self.backbone(x, training=training)
        return self.dense(x)

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
        return input_shape[0], self.desc_model.embed_dim


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

    def infer_initial_state(self, profile, batch_size=32, **kwargs):
        return self.profile_model.predict(
            profile, batch_size=batch_size, **kwargs)


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
        'image': layers.Input(shape=(h, w, 3), dtype=tf.uint8),
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

    def __init__(self, config, item_model, user_model, item_data, **kwargs):
        super(RecModel, self).__init__(**kwargs)
        self.config = config
        self.item_model = item_model
        self.user_model = user_model
        # Cache item data to accelarate
        self.item_data = item_data

    def compile(self, optimizer, margin=0.0, gamma=1.0):
        super(RecModel, self).compile(optimizer=optimizer)
        self.loss_fn = UnifiedLoss(
            margin=margin, gamma=gamma,
            reduction=tf.keras.losses.Reduction.NONE
        )

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            batch_size = tf.shape(inputs['items'])[0]
            seq_length = tf.shape(inputs['items'])[1]
            # compute item vectors
            item_indices = tf.reshape(inputs['items'], [-1])
            pad_mask = tf.not_equal(item_indices, -1)
            item_vectors = self.item_model(
                {
                    'info': tf.gather(self.item_data['info'], item_indices),
                    'desc': tf.gather(self.item_data['desc'], item_indices),
                    'image': tf.gather(self.item_data['image'], item_indices)
                },
                training=True
            )
            item_vectors *= tf.expand_dims(tf.cast(pad_mask, tf.float32), -1)
            item_vectors = tf.reshape(item_vectors, [batch_size, seq_length, -1])

            # compute user vectors
            state_seq, _ = self.user_model(
                {
                    'profile': inputs['profile'],
                    'context': inputs['context'],
                    'items': item_vectors
                },
                training=True
            )

            batch_idx = tf.range(0, batch_size)
            length_idx = tf.range(0, seq_length)
            a = batch_idx[:, tf.newaxis, tf.newaxis, tf.newaxis]
            b = length_idx[tf.newaxis, :, tf.newaxis, tf.newaxis]
            c = batch_idx[tf.newaxis, tf.newaxis, :, tf.newaxis]
            d = length_idx[tf.newaxis, tf.newaxis, tf.newaxis, :]

            # mask history items and items out of prediction length
            prd_mask = tf.logical_and(
                tf.equal(a, c),
                tf.logical_or(
                    tf.greater_equal(b, d), tf.greater(d-b, self.config['predict_length']))
            )
            prd_mask = tf.reshape(prd_mask, [batch_size, seq_length, -1])  # (batch, len, batch * len)
            prd_mask = tf.logical_not(prd_mask)
            pad_mask = pad_mask[tf.newaxis, tf.newaxis, :]

            # mask same items
            items_a = inputs['items'][:, :, tf.newaxis, tf.newaxis]
            items_b = inputs['items'][tf.newaxis, tf.newaxis, :, :]
            same_mask = tf.not_equal(items_a, items_b)
            same_mask = tf.reshape(same_mask, [batch_size, seq_length, -1])  # (batch, len, batch * len)

            # (batch, len, batch * len)
            mask = tf.logical_and(tf.logical_and(pad_mask, prd_mask), same_mask)

            # compute logits
            item_vectors = tf.reshape(item_vectors, [-1, self.config['embed_dim']])  # (batch * len, dim)
            logits = tf.matmul(state_seq, item_vectors, transpose_b=True)  # (batch, len, batch * len)

            # compute labels
            labels = tf.tile(tf.equal(a, c), [1, seq_length, 1, seq_length])
            labels = tf.cast(tf.reshape(labels, [batch_size, seq_length, -1]), tf.float32)
            labels = tf.cast(tf.where(mask, labels, -1), labels.dtype)

            loss = self.loss_fn(labels, logits)

        variables = self.item_model.trainable_weights+self.user_model.trainable_weights
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'loss': loss}

    def save_weights(self, filepath, **kwargs):
        self.item_model.save_weights(os.path.join(filepath, 'item.h5'), **kwargs)
        self.user_model.save_weights(os.path.join(filepath, 'user.h5'), **kwargs)


class RecInfer(tf.keras.Model):
    def __init__(self, user_model, item_vectors,
                 top_k=10, skip_used_items=False, **kwargs):
        super(RecInfer, self).__init__(**kwargs)
        self.user_model = user_model
        self.item_vectors = item_vectors
        self.top_k = top_k
        self.skip_used_items = skip_used_items

        dummy_inputs = {
            'profile': layers.Input(shape=user_model.input_shape['profile'][1:], dtype=tf.int32),
            'context': layers.Input(shape=user_model.input_shape['context'][1:], dtype=tf.int32),
            'item_indices': layers.Input(shape=user_model.input_shape['items'][1:-1], dtype=tf.int32)
        }
        self(dummy_inputs)

    def call(self, inputs):
        batch_size = tf.shape(inputs['item_indices'])[0]
        seq_length = tf.shape(inputs['item_indices'])[1]
        item_indices = tf.reshape(inputs.pop('item_indices'), [-1])
        inputs['items'] = tf.reshape(
            tf.gather(self.item_vectors, item_indices),
            [batch_size, seq_length, -1]
        )
        _, user_vector = self.user_model(inputs, training=False)
        score = tf.matmul(user_vector, self.item_vectors, transpose_b=True)
        if self.skip_used_items:
            # mask used_items
            used_items = tf.reshape(item_indices, [batch_size, seq_length])
            item_size = tf.shape(self.item_vectors)[0]
            used_items = tf.reshape(used_items, [-1])
            mask = tf.one_hot(used_items, depth=item_size, dtype=tf.int8)
            mask = tf.reshape(mask, [batch_size, -1, item_size])
            mask = tf.reduce_any(tf.not_equal(mask, 0), axis=1)
            score -= 1e5*tf.cast(mask, score.dtype)

        recommend = tf.argsort(score, direction='DESCENDING')[:, :self.top_k]
        return recommend


if __name__ == '__main__':
    import numpy as np
    from evaluate import MAP

    config = {
        'max_history_length': 16,
        'predict_length': 12,
        'max_desc_length': 8,
        'info_size': [4, 5, 6],
        'profile_size': [2, 3],
        'context_size': [3, 5],
        'embed_dim': 64,
        'bert_path': 'bert-base-uncased',
        'image_weights': 'imagenet',
        'image_height': 32,
        'image_width': 32
    }

    item_data = {
        'info': np.random.randint(0, 4, (10, 3)),
        'desc': np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]]*10),
        'image': np.random.randint(0, 255, size=(10, 32, 32, 3))
    }

    item_model, user_model = build_model(config)
    item_model.summary()
    user_model.summary()
    rec_model = RecModel(config, item_model, user_model, item_data)
    rec_model.compile('adam')

    for w in item_model.trainable_weights:
        print(w.name)

    print('\n\n')
    for w in user_model.trainable_weights:
        print(w.name)

    # inputs = {
    #     'items': tf.constant([[1, 3], [2, 5]], dtype=tf.int32),
    #     'profile': tf.constant([[1, 2], [0, 2]], dtype=tf.int32),
    #     'context': tf.constant([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], dtype=tf.int32)
    # }
    # loss = rec_model.train_step(inputs)

    # item_vectors = tf.identity(item_model.predict(item_data))
    # rec_infer = RecInfer(user_model, item_vectors, top_k=5)
    # rec_infer.compile(metrics=[MAP(5)])

    # inputs['item_indices'] = inputs.pop('items')
    # ground_truth = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
    # print(rec_infer.predict(inputs))
    # results = rec_infer.evaluate(inputs, ground_truth)
    # print(results)
