import numpy as np
from threading import Thread
from multiprocessing.pool import ThreadPool
from collections import deque
from typing import Union
from cv2 import cv2 as cv
import time
from colorama import Fore
import random


class AugmentationSettings:
  def __init__(self, rotation_chance:float=0, rotation_ammount:float=0, blur_chance:float=0, blur_amount:float=0, flip_chance:float=0):
    self.rotation_chance = rotation_chance
    self.rotation_ammount = rotation_ammount
    self.blur_chance = blur_chance
    self.blur_amount = blur_amount
    self.flip_chance = flip_chance

class BatchMaker(Thread):
  def __init__(self, train_data:list, batch_size:int, buffered_batches:int=5, secondary_size:tuple=None, missing_threshold_perc:float=0.2, num_of_loading_workers:int=8, augmentation_settings:AugmentationSettings=None):
    super().__init__()
    self.daemon = True

    self.__terminate = False

    self.__secondary_size = secondary_size
    self.augmentation_settings = augmentation_settings

    self.__batches_in_buffer_number = buffered_batches
    assert 0 <= missing_threshold_perc <= 1, Fore.RED + "Invalid missing threshold" + Fore.RESET
    self.__missing_threshold_number = int(self.__batches_in_buffer_number * missing_threshold_perc)
    self.__batches = deque(maxlen=self.__batches_in_buffer_number)
    self.__resized_batches = deque(maxlen=self.__batches_in_buffer_number)

    self.__train_data = train_data
    self.__data_length = len(self.__train_data)
    assert self.__data_length > 0, Fore.RED + "Dataset is empty" + Fore.RESET

    self.__batch_size = batch_size

    self.__index = 0
    self.__max_index = (self.__data_length // self.__batch_size) - 2
    self.__worker_pool = ThreadPool(processes=num_of_loading_workers)

    self.__lock = False
    self.__lock_confirm = False

    self.start()

  def terminate(self):
    self.__terminate = True

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

        if self.augmentation_settings:
          if random.random() >= self.augmentation_settings.blur_chance:
            original_image = cv.GaussianBlur(original_image, (3, 3), self.augmentation_settings.blur_amount)

          if random.random() >= self.augmentation_settings.flip_chance:
            original_image = cv.flip(original_image, random.randint(-1, 1))

          if random.random() >= self.augmentation_settings.rotation_chance:
            rows, cols, c = original_image.shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), random.random() * self.augmentation_settings.rotation_ammount, 1)
            original_image = cv.warpAffine(original_image, M, (cols, rows))

        batch.append(cv.cvtColor(original_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0)
        if self.__secondary_size:
          resized_batch.append(cv.cvtColor(cv.resize(original_image, dsize=(self.__secondary_size[0], self.__secondary_size[1]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > self.__secondary_size[1] and original_image.shape[1] > self.__secondary_size[0]) else cv.INTER_CUBIC)), cv.COLOR_BGR2RGB) / 127.5 - 1.0)

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

    while not self.__terminate:
      while self.__lock:
        if not self.__lock_confirm:
          self.__lock_confirm = True
        time.sleep(0.001)

      if self.__lock_confirm:
        self.__lock_confirm = False

      batches_to_create = self.__batches_in_buffer_number - len(self.__batches)
      if batches_to_create > self.__missing_threshold_number:
        self.__worker_pool.map(self.__make_batch, self.__make_data(batches_to_create))

      time.sleep(0.01)

    self.__worker_pool.close()

  def get_batch(self) -> Union[np.ndarray, tuple]:
    while not self.__batches: time.sleep(0.01)
    if self.__secondary_size:
      while not self.__resized_batches: time.sleep(0.01)
      return self.__batches.popleft(), self.__resized_batches.popleft()
    return self.__batches.popleft()