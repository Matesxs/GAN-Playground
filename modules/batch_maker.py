from typing import Union
import numpy as np
from threading import Thread
from collections import deque
import cv2 as cv
import time

class BatchMaker(Thread):
	def __init__(self, train_data: Union[list, np.ndarray], data_length: int, batch_size: int, buffered_batches:int=5):
		super().__init__()
		self.daemon = True

		self.terminate = False

		self.batches_in_buffer = buffered_batches
		self.batches = deque(maxlen=self.batches_in_buffer)

		self.train_data = train_data
		self.data_length = data_length
		self.batch_size = batch_size

	def run(self):
		while not self.terminate:
			if len(self.batches) < self.batches_in_buffer:
				if type(self.train_data) == list:
					# Load and normalize images if train_data is list of paths
					self.batches.append(np.array([cv.cvtColor(cv.imread(im_p), cv.COLOR_BGR2RGB) / 127.5 - 1.0 for im_p in np.array(self.train_data)[np.random.randint(0, self.data_length, self.batch_size)]]).astype(np.float32))
				else:
					self.batches.append(np.array(self.train_data[np.random.randint(0, self.data_length, self.batch_size)] / 127.5 - 1.0).astype(np.float32))
			time.sleep(0.01)

	def get_batch(self) -> np.ndarray:
		while not self.batches: time.sleep(0.02)
		return self.batches.pop()