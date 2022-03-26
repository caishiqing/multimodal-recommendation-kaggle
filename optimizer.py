import tensorflow as tf

__all__ = [
    'WarmUpSchedule', 'AdamWarmup',
]


class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(
            self,
            initial_learning_rate,
            decay_schedule_fn,
            warmup_steps,
            power=1.0,
            name=None):
        super(WarmUpSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmUp') as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = (
                self.initial_learning_rate *
                tf.math.pow(warmup_percent_done, self.power))
            return tf.cond(global_step_float < warmup_steps_float,
                           lambda: warmup_learning_rate,
                           lambda: self.decay_schedule_fn(step),
                           name=name)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_schedule_fn': self.decay_schedule_fn,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
            'name': self.name
        }


class AdamWarmup(tf.keras.optimizers.Adam):
    def __init__(self, warmup_steps, decay_steps,
                 initial_learning_rate=1e-3,
                 end_learning_rate=1e-6,
                 **kwargs):

        decay_schedule_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate, decay_steps, end_learning_rate,
        )
        learning_schedule_fn = WarmUpSchedule(
            initial_learning_rate, decay_schedule_fn, warmup_steps,
        )
        kwargs['learning_rate'] = learning_schedule_fn
        super(AdamWarmup, self).__init__(**kwargs)
