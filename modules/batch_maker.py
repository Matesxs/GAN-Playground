import numpy as np
from threading import Thread
from multiprocessing.pool import ThreadPool
from collections import deque
from typing import Union
import cv2 as cv
import time
from colorama import Fore

from settings import NUM_OF_LOADING_WORKERS

class BatchMaker(Thread):
  def __init__(self, train_data:list, batch_size: int, buffered_batches:int=5, secondary_size:tuple=None):
    super().__init__()
    self.daemon = True

    self.terminate = False
    self.__secondary_size = secondary_size

    self.__batches_in_buffer_number = buffered_batches
    self.__batches = deque(maxlen=self.__batches_in_buffer_number)
    self.__resized_batches = deque(maxlen=self.__batches_in_buffer_number)

    self.__train_data = train_data
    self.__data_length = len(self.__train_data)
    assert self.__data_length > 0, Fore.RED + "Dataset is empty" + Fore.RESET

    self.__batch_size = batch_size

    self.__index = 0
    self.__max_index = (self.__data_length // self.__batch_size) - 2
    self.__worker_pool = ThreadPool(processes=NUM_OF_LOADING_WORKERS)

    self.__lock = False
    self.__lock_confirm = False

  def get_number_of_batches_in_dataset(self):
    return self.__max_index

  def reset_stored_batches(self):
    self.__lock = True
    while not self.__lock_confirm:
      time.sleep(0.001)

    self.__index = 0
    self.__batches.clear()
    self.__resized_batches.clear()
    self.__lock = False

    while self.__lock_confirm:
      time.sleep(0.001)

  def __make_batch(self, data):
    if data is not None:
      batch = []
      resized_batch = []

      for im_p in data:
        original_image = cv.imread(im_p)
        batch.append(cv.cvtColor(original_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0)
        if self.__secondary_size:
          resized_batch.append(cv.cvtColor(cv.resize(original_image, dsize=(self.__secondary_size[0], self.__secondary_size[1]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > self.__secondary_size[0] and original_image.shape[1] > self.__secondary_size[1]) else cv.INTER_CUBIC)), cv.COLOR_BGR2RGB) / 127.5 - 1.0)

      if batch: self.__batches.append(np.array(batch).astype(np.float32))
      if resized_batch: self.__resized_batches.append(np.array(resized_batch).astype(np.float32))

  def __make_data(self, data_ammount:int):
    data_array = []
    for _ in range(data_ammount):
      data_array.append(np.array(self.__train_data)[range(self.__index * self.__batch_size, (self.__index + 1) * self.__batch_size)])

      self.__index += 1
      if self.__index >= self.__max_index:
        np.random.shuffle(self.__train_data)
        self.__index = 0

    return data_array

  def run(self):
    np.random.shuffle(self.__train_data)

    while not self.terminate:
      while self.__lock:
        if not self.__lock_confirm:
          self.__lock_confirm = True
        time.sleep(0.001)

      if self.__lock_confirm:
        self.__lock_confirm = False

      batches_to_create = self.__batches_in_buffer_number - len(self.__batches)
      if batches_to_create > 0:
        self.__worker_pool.map(self.__make_batch, self.__make_data(batches_to_create))

      time.sleep(0.01)

    self.__worker_pool.close()

  def get_batch(self) -> Union[np.ndarray, tuple]:
    while not self.__batches: time.sleep(0.01)
    if self.__secondary_size:
      while not self.__resized_batches: time.sleep(0.01)
      return self.__batches.popleft(), self.__resized_batches.popleft()
    return self.__batches.popleft()