from threading import Thread
from modules.keras_extensions.custom_tensorboard import TensorBoardCustom
from collections import deque
import time

class StatLogger(Thread):
  def __init__(self, tensorboard:TensorBoardCustom, write_threshold:int=50):
    super(StatLogger, self).__init__()
    self.daemon = True

    self.__tensorboard = tensorboard
    self.__write_threshold = write_threshold
    self.__stats = deque()

    self.__terminated = False

    self.start()

  def terminate(self):
    self.__terminated = True

  def append_stats(self, step, **stats):
    self.__stats.append([step, stats])

  def run(self) -> None:
    while not self.__terminated:
      len_of_stats = len(self.__stats)
      if len_of_stats > self.__write_threshold:
        for _ in range(len_of_stats):
          stat_pair = self.__stats.popleft()
          self.__tensorboard._write_logs(stat_pair[1], stat_pair[0])
      time.sleep(0.05)

    len_of_stats = len(self.__stats)
    if len_of_stats > 0:
      for _ in range(len_of_stats):
        stat_pair = self.__stats.popleft()
        self.__tensorboard._write_logs(stat_pair[1], stat_pair[0])