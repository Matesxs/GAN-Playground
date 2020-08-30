from keras.optimizers import Optimizer
from keras.legacy import interfaces
import keras.backend as K
import tensorflow as tf

class AccumulateAdam(Optimizer):
  def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, accum_iters:int=1, **kwargs):
    assert accum_iters > 0, "accum_iters must be >= 1"
    self.initial_decay = kwargs.pop('decay', 0.0)
    self.epsilon = kwargs.pop('epsilon', K.epsilon())
    learning_rate = kwargs.pop('lr', learning_rate)
    super(AccumulateAdam, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.learning_rate = K.variable(learning_rate, name='learning_rate')
      self.beta_1 = K.variable(beta_1, name='beta_1')
      self.beta_2 = K.variable(beta_2, name='beta_2')
      self.decay = K.variable(self.initial_decay, name='decay')
    self.amsgrad = amsgrad
    self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
    self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

  @interfaces.legacy_get_updates_support
  @K.symbolic
  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.learning_rate

    completed_updates = K.cast(tf.math.floordiv(self.iterations, self.accum_iters), K.floatx())

    if self.initial_decay > 0:
      lr = lr * (1. / (1. + self.decay * completed_updates))

    t = completed_updates + 1

    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

    # self.iterations incremented after processing a batch
    # batch:              1 2 3 4 5 6 7 8 9
    # self.iterations:    0 1 2 3 4 5 6 7 8
    # update_switch = 1:        x       x    (if accum_iters=4)
    update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
    update_switch = K.cast(update_switch, K.floatx())

    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

    if self.amsgrad:
      vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    else:
      vhats = [K.zeros(1) for _ in params]

    self.weights = [self.iterations] + ms + vs + vhats

    for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

      sum_grad = tg + g
      avg_grad = sum_grad / self.accum_iters_float

      m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

      if self.amsgrad:
        vhat_t = K.maximum(vhat, v_t)
        p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
        self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
      else:
        p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

      self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
      self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
      self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
    return self.updates

  def get_config(self):
    config = {'learning_rate': float(K.get_value(self.learning_rate)),
              'beta_1': float(K.get_value(self.beta_1)),
              'beta_2': float(K.get_value(self.beta_2)),
              'decay': float(K.get_value(self.decay)),
              'epsilon': self.epsilon,
              'amsgrad': self.amsgrad}
    base_config = super(AccumulateAdam, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
