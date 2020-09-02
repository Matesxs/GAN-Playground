import keras.backend as K
from typing import Union
from keras import Model
from keras.optimizers import Optimizer

class LearningRateScheduler:
  """Learning rate scheduler."""

  def __init__(self, start_lr:Union[float, None], lr_decay_factor:Union[float, None], lr_decay_interval:Union[int, None], min_lr:Union[float, None]=1e-7):
    self.start_lr = start_lr
    self.lr_decay_factor = lr_decay_factor
    self.lr_decay_interval = lr_decay_interval
    self.min_lr = min_lr

    if self.min_lr is None: self.min_lr = 0

    assert self.min_lr >= 0, "Invalid value of min lr"
    if self.lr_decay_factor:
      assert 1 >= self.lr_decay_factor >= 0, "Decay factor must be 0 <= decay factor <= 1"
    if self.lr_decay_interval:
      assert self.lr_decay_interval >= 0, "Decay interval must be >= 0"

    self.current_lr = self.start_lr

  def set_lr(self, object:Union[Model, Optimizer], episode:int):
    if self.lr_decay_factor is None or self.lr_decay_interval is None or self.start_lr is None: return None
    if self.lr_decay_factor == 0 or self.lr_decay_interval == 0: return None

    n_decays = episode // self.lr_decay_interval
    lr = self.start_lr * (self.lr_decay_factor ** n_decays)
    lr = max(lr, self.min_lr)

    if isinstance(object, Model):
      if not hasattr(object.optimizer, 'lr'):
        raise ValueError('Optimizer must have a "lr" attribute.')

      if float(K.get_value(object.optimizer.lr)) != lr:
        K.set_value(object.optimizer.lr, lr)
        self.current_lr = lr
        return lr

    elif isinstance(object, Optimizer):
      if float(K.get_value(object.lr)) != lr:
        K.set_value(object.lr, lr)
        self.current_lr = lr
        return lr

    return None