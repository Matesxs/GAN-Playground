import keras.backend as K
from typing import Union
from keras import Model
from keras.optimizers import Optimizer

class LearningRateScheduler:
  """Learning rate scheduler."""

  def __init__(self, lr_plan:Union[dict, None], start_lr:Union[float, None]=None):
    self.lr_plan = lr_plan

    self.episode = 0
    self.lr = start_lr

  def set_lr(self, object):
    if isinstance(object, Model):
      if not hasattr(object.optimizer, 'lr'):
        raise ValueError('Optimizer must have a "lr" attribute.')

      if self.lr:
        if float(K.get_value(object.optimizer.lr)) != self.lr:
          K.set_value(object.optimizer.lr, self.lr)

      if not self.lr_plan: return None
      if not self.episode in self.lr_plan: return None

      new_lr = self.lr_plan[self.episode]
      K.set_value(object.optimizer.lr, new_lr)
      self.lr = new_lr
      return new_lr
    elif isinstance(object, Optimizer):
      if self.lr:
        if float(K.get_value(object.lr)) != self.lr:
          K.set_value(object.lr, self.lr)

      if not self.lr_plan: return None
      if not self.episode in self.lr_plan: return None

      new_lr = self.lr_plan[self.episode]
      K.set_value(object.lr, new_lr)
      self.lr = new_lr
      return new_lr