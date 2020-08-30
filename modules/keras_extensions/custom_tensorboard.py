import keras
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import os

# Based on ModifiedTensorBoard from Sendex (https://pythonprogramming.net)
# https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/ (original code)

class TensorBoardCustom(Callback):
  def __init__(self, log_dir):
    super().__init__()
    self.step = 0
    self.__log_dir = log_dir
    self.__writer = None

    if not os.path.exists(self.__log_dir): os.makedirs(self.__log_dir)

  def __del__(self):
    try:
      if self.__writer:
        self.__writer.close()
    except:
      pass

  def on_epoch_end(self, epoch, logs=None):
    pass

  def init_writer_check(self):
    if not self.__writer:
      self.__writer = tf.summary.create_file_writer(self.__log_dir)

  def log_kernels_and_biases(self, model:keras.Model):
    self.init_writer_check()

    with self.__writer.as_default():
      for layer in model.layers:
        for weight in layer.weights:
          weight_name = weight.name.replace(':', '_')
          weight = K.get_value(weight)
          tf.summary.histogram(weight_name, weight, step=self.step)
    self.__writer.flush()

  def update_stats(self, **stats):
    self._write_logs(stats, self.step)

  # More or less the same writer as in Keras' Tensorboard callback
  # Physically writes to the log files
  def _write_logs(self, logs, index):
    self.init_writer_check()

    with self.__writer.as_default():
      for name, value in logs.items():
        if name in ['batch', 'size']:
          continue

        tf.summary.scalar(name, value, step=index)
    self.__writer.flush()

  def write_image(self, image:np.ndarray, description:str="progress"):
    self.init_writer_check()

    with self.__writer.as_default():
      tf.summary.image(description, image, step=self.step)
    self.__writer.flush()