from tensorflow.python.training import gen_training_ops
import tensorflow as tf
import re

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
                 lr_multiply=None,
                 **kwargs):

        decay_schedule_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate, decay_steps, end_learning_rate,
        )
        learning_schedule_fn = WarmUpSchedule(
            initial_learning_rate, decay_schedule_fn, warmup_steps,
        )
        kwargs['learning_rate'] = learning_schedule_fn
        super(AdamWarmup, self).__init__(**kwargs)

        # {regexp: x.x}
        self.lr_multiply = lr_multiply if lr_multiply else {}

    def _resource_apply_dense(self, grad, var, apply_state=None):
        multiply = 1
        for reg, mult in self.lr_multiply.items():
            if re.search(reg, var.name):
                multiply *= mult
                break

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        if not self.amsgrad:
            return gen_training_ops.ResourceApplyAdam(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=coefficients['beta_1_power'],
                beta2_power=coefficients['beta_2_power'],
                lr=coefficients['lr_t'] * multiply,
                beta1=coefficients['beta_1_t'],
                beta2=coefficients['beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)
        else:
            vhat = self.get_slot(var, 'vhat')
            return gen_training_ops.ResourceApplyAdamWithAmsgrad(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                vhat=vhat.handle,
                beta1_power=coefficients['beta_1_power'],
                beta2_power=coefficients['beta_2_power'],
                lr=coefficients['lr_t'] * multiply,
                beta1=coefficients['beta_1_t'],
                beta2=coefficients['beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)

    # def _resource_apply_dense(self, grad, var, apply_state=None):
    #     # Different lr for defferent variables
    #     var_device, var_dtype = var.device, var.dtype.base_dtype
    #     if not apply_state:
    #         apply_state = {(var_device, var_dtype): self._fallback_apply_state(var_device, var_dtype)}

    #     if self.lr_multiply:
    #         for regexp, multiply in self.lr_multiply.items():
    #             if re.search(regexp, var.name):
    #                 if (var_device, var_dtype) not in apply_state:
    #                     key = list(apply_state)[0]
    #                 else:
    #                     key = (var_device, var_dtype)
    #                 apply_state[key]['lr_t'] *= multiply
    #                 break

    #     return super(AdamWarmup, self)._resource_apply_dense(grad, var, apply_state=apply_state)
