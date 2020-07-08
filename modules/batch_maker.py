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
    self.secondary_size = secondary_size

    self.batches_in_buffer = buffered_batches
    self.batches = deque(maxlen=self.batches_in_buffer)
    self.resized_batches = deque(maxlen=self.batches_in_buffer)

    self.train_data = train_data
    self.data_length = len(self.train_data)
    assert self.data_length > 0, Fore.RED + "Dataset is empty" + Fore.RESET

    self.batch_size = batch_size

    self.index = 0
    self.max_index = (self.data_length // self.batch_size) - 2
    self.worker_pool = ThreadPool(processes=NUM_OF_LOADING_WORKERS)

  def make_batch(self, data):
    if data is not None:
      batch = []
      resized_batch = []

      for im_p in data:
        original_image = cv.imread(im_p)
        batch.append(cv.cvtColor(original_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0)
        if self.secondary_size:
          resized_batch.append(cv.cvtColor(cv.resize(original_image, dsize=(self.secondary_size[0], self.secondary_size[1]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > self.secondary_size[0] and original_image.shape[1] > self.secondary_size[1]) else cv.INTER_CUBIC)), cv.COLOR_BGR2RGB) / 127.5 - 1.0)

      if batch: self.batches.append(np.array(batch).astype(np.float32))
      if resized_batch: self.resized_batches.append(np.array(resized_batch).astype(np.float32))

  def make_data(self, data_ammount:int):
    data_array = []
    for _ in range(data_ammount):
      data_array.append(np.array(self.train_data)[range(self.index * self.batch_size, (self.index + 1) * self.batch_size)])

      self.index += 1
      if self.index >= self.max_index:
        np.random.shuffle(self.train_data)
        self.index = 0

    return data_array

  def run(self):
    np.random.shuffle(self.train_data)

    while not self.terminate:
      batches_to_create = self.batches_in_buffer - len(self.batches)
      if batches_to_create > 0:
        self.worker_pool.map(self.make_batch, self.make_data(batches_to_create))

      time.sleep(0.01)

  def get_batch(self) -> Union[np.ndarray, tuple]:
    while not self.batches: time.sleep(0.01)
    if self.secondary_size:
      while not self.resized_batches: time.sleep(0.01)
      return self.batches.popleft(), self.resized_batches.popleft()
    return self.batches.popleft()