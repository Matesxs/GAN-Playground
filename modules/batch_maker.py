import numpy as np
from threading import Thread
from collections import deque
import cv2 as cv
import time

# TODO: Add workers to loading batches for faster batch creation
class BatchMaker(Thread):
	def __init__(self, train_data:list, data_length: int, batch_size: int, buffered_batches:int=5):
		super().__init__()
		self.daemon = True

		self.terminate = False

		self.batches_in_buffer = buffered_batches
		self.batches = deque(maxlen=self.batches_in_buffer)

		self.train_data = train_data
		self.data_length = data_length
		self.batch_size = batch_size

		self.max_index = (self.data_length // self.batch_size) - 2

	def run(self):
		index = 0
		np.random.shuffle(self.train_data)

		while not self.terminate:
			if len(self.batches) < self.batches_in_buffer:
				# Load and normalize images if train_data is list of paths
				self.batches.append(np.array([cv.cvtColor(cv.imread(im_p), cv.COLOR_BGR2RGB) / 127.5 - 1.0 for im_p in np.array(self.train_data)[range(index * self.batch_size, (index + 1) * self.batch_size)]]).astype(np.float32))

				index += 1
				if index >= self.max_index:
					np.random.shuffle(self.train_data)
					index = 0

			time.sleep(0.01)

	def get_batch(self) -> np.ndarray:
		while not self.batches: time.sleep(0.01)
		return self.batches.popleft()