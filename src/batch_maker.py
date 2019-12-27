from typing import Union
import numpy as np
from threading import Thread
import cv2 as cv
import time
from copy import copy

class BatchMaker(Thread):
	def __init__(self, train_data: Union[list, np.ndarray], data_length: int, batch_size: int):
		super().__init__()
		self.daemon = True

		self.terminate = False

		self.batch_ready = False
		self.batch = None

		self.train_data = train_data
		self.data_length = data_length
		self.batch_size = batch_size

	def run(self):
		while not self.terminate:
			if not self.batch_ready:
				if type(self.train_data) == list:
					# Load and normalize images if train_data is list of paths
					self.batch = np.array([cv.imread(im_p) / 127.5 - 1.0 for im_p in np.array(self.train_data)[np.random.randint(0, self.data_length, self.batch_size)]])
				else:
					self.batch = self.train_data[np.random.randint(0, self.data_length, self.batch_size)]
				self.batch_ready = True
			time.sleep(0.02)

	def get_batch(self):
		while not self.batch_ready: time.sleep(0.02)
		tmp_batch = copy(self.batch)
		self.batch_ready = False
		return tmp_batch