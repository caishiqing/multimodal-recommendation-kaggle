import tensorflow as tf
import numpy as np


class UniteLoss(tf.losses.Loss):
    """ 统一视角下的 loss """

    def __init__(self, margin=0.25, gamma=32, **kwargs):
        """
        Args:
            margin (float, optional): 间隙
            gamma (float, optional): 缩放尺度
        """
        super(UniteLoss, self).__init__(**kwargs)
        assert 0 <= margin < 1
        assert gamma > 0

        self.margin = margin
        self.gamma = gamma

    def call(self, y_true, y_pred):
        pos_mask = tf.cast(y_true, tf.float32)
        neg_mask = 1 - pos_mask

        logit_p = -y_pred * self.gamma * pos_mask - neg_mask * 1e7
        logit_n = (y_pred + self.margin) * self.gamma * neg_mask - pos_mask * 1e7

        loss = tf.math.log(
            1 + tf.reduce_sum(tf.exp(logit_p), -1) *
            tf.reduce_sum(tf.exp(logit_n), -1)
        )
        loss = tf.nn.softplus(tf.reduce_logsumexp(logit_p, -1) + tf.reduce_logsumexp(logit_n, -1))
        return tf.reduce_mean(loss)


class CircleLoss(UniteLoss):
    """ Circle Loss: A Unified Perspective of Pair Similarity Optimization - CVPR2020 """

    def __init__(self, **kwargs):
        # margin 代表松弛因子
        super(CircleLoss, self).__init__(**kwargs)
        self.opt_p = 1 + self.margin
        self.opt_n = - self.margin
        self.delta_p = 1 - self.margin
        self.delta_n = self.margin

    def call(self, y_true, y_pred):
        # y_pred 为归一化度量, 0 <= y_pred <= 1
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, y_pred.dtype)

        pos_index = tf.where(y_true > 0)
        neg_index = tf.where((1 - y_true) > 0)

        sp = tf.gather_nd(y_pred, pos_index)
        sn = tf.gather_nd(y_pred, neg_index)

        alpha_p = tf.nn.relu(self.opt_p - tf.stop_gradient(sp))
        alpha_n = tf.nn.relu(tf.stop_gradient(sn) - self.opt_n)

        logit_p = alpha_p * (self.delta_p - sp) * self.gamma
        logit_n = alpha_n * (sn - self.delta_n) * self.gamma
        # softplus(x) = log(1 + exp(x))
        loss = tf.nn.softplus(tf.reduce_logsumexp(logit_n, axis=-1) +
                              tf.reduce_logsumexp(logit_p, axis=-1))

        return loss


class MAP:
    def __init__(self, top_k=10):
        self.top_k = top_k

    def __call__(self, ground_truth, predictions):
        ground_truth = tf.keras.preprocessing.sequence.pad_sequences(
            ground_truth, maxlen=self.top_k, value=-1)
        m = np.sum(ground_truth != -1, axis=1)

        # Exclude no-ground_truth samples
        valid_index = np.where(m > 0)
        m = m[valid_index]
        x = predictions[:, :self.top_k][valid_index]

        a = np.expand_dims(ground_truth, 2)
        b = np.expand_dims(x, 1)
        c = a == b

        rel = np.any(c, axis=1)
        pre = np.cumsum(rel, axis=-1, dtype=np.float32) / np.array([range(1, self.top_k+1)], np.float32)
        ap = np.sum(pre*rel, axis=1)/m
        map = np.mean(ap)

        return map
