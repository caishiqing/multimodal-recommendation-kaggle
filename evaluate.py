import tensorflow as tf
import numpy as np


class UnifiedLoss(tf.keras.losses.Loss):
    """ From Circle Loss (CVPR 2020) """

    def __init__(self, margin=0, gamma=1, **kwargs):
        """
        Args:
            margin (float, optional): Margin between positive and negtive
            gamma (float, optional): Multiplier of logits
        """
        super(UnifiedLoss, self).__init__(**kwargs)
        assert 0 <= margin < 1
        assert gamma > 0

        self.margin = margin
        self.gamma = gamma

    def call(self, y_true, y_pred):
        pos_index = tf.where(y_true == 1)
        neg_index = tf.where(y_true == 0)

        sp = tf.gather_nd(y_pred, pos_index)
        sn = tf.gather_nd(y_pred, neg_index)

        loss = tf.math.log(
            1 + tf.reduce_sum(tf.math.exp(-self.gamma * sp)) *
            tf.reduce_sum(tf.math.exp(self.gamma * (sn + self.margin)))
        )
        return loss


class CircleLoss(UnifiedLoss):
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


class MAP(tf.keras.metrics.Mean):
    def __init__(self, top_k=10, **kwargs):
        kwargs['name'] = f'MAP@{top_k}'
        kwargs['dtype'] = tf.float32
        super(MAP, self).__init__(**kwargs)
        self.top_k = top_k

    def update_state(self, ground_truth, predictions, sample_weight=None):
        # cut off top_k
        ground_truth = tf.cast(ground_truth, self._dtype)[:, :self.top_k]
        predictions = tf.cast(predictions, self._dtype)[:, :self.top_k]

        # ground_truth length, mask samples no ground_truth
        m = tf.reduce_sum(tf.cast(tf.not_equal(ground_truth, -1), tf.float32), axis=-1)
        sample_weight = tf.where(tf.equal(m, 0), 0, 1)
        m = tf.where(tf.equal(m, 0), 1.0, m)

        # mask predictions
        mask = tf.not_equal(predictions, -1)
        mask = tf.cast(mask, tf.float32)

        a = tf.expand_dims(ground_truth, 2)
        b = tf.expand_dims(predictions, 1)
        indicate = tf.equal(a, b)

        # indicator equaling 1 if the predicted item at rank k is a relevant (correct) label, 0 otherwise
        rel = tf.cast(tf.reduce_any(indicate, 1), tf.float32)
        pred_len = tf.range(1, self.top_k+1, dtype=tf.float32)[tf.newaxis, :]
        # precision of cut off k
        pre = tf.cumsum(rel, axis=-1) / pred_len
        map = tf.reduce_sum(pre*rel*mask, axis=-1) / m

        return super(MAP, self).update_state(map, sample_weight=sample_weight)


if __name__ == '__main__':
    ground_truth = [[1, 2, 3, 4, 5], [6, 7, 8, -1, -1], [-1, -1, -1, -1, -1]]
    predictions = [[3, 6, 7, 2, 4], [6, 4, 1, 6, -1], [1, 6, 3, 4, 5]]

    map = MAP(5)
    map.update_state(ground_truth, predictions)
    print(map.result().numpy())
