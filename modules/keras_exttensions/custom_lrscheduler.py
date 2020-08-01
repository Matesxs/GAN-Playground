import keras.backend as K
from typing import Union

class LearningRateScheduler:
  """Learning rate scheduler."""

  def __init__(self, lr_plan:Union[dict, None], start_lr:Union[float, None]=None):
    self.lr_plan = lr_plan

    self.episode = 0
    self.lr = start_lr

  def set_lr(self, model):
    if not hasattr(model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')

    if self.lr:
      if float(K.get_value(model.optimizer.lr)) != self.lr:
        K.set_value(model.optimizer.lr, self.lr)

    if not self.lr_plan: return None
    if not self.episode in self.lr_plan: return None

    new_lr = self.lr_plan[self.episode]
    K.set_value(model.optimizer.lr, new_lr)
    self.lr = new_lr
    return new_lr