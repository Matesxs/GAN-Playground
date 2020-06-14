import numpy as np
from threading import Thread
from multiprocessing.pool import ThreadPool
from collections import deque
import cv2 as cv
import time

# TODO: Add workers to loading batches for faster batch creation
class BatchMaker(Thread):
	def __init__(self, train_data:list, data_length: int, batch_size: int, buffered_batches:int=5, num_of_workers:int=4):
		super().__init__()
		self.daemon = True

		self.terminate = False

		self.batches_in_buffer = buffered_batches
		self.batches = deque(maxlen=self.batches_in_buffer)

		self.train_data = train_data
		self.data_length = data_length
		self.batch_size = batch_size

		self.index = 0
		self.max_index = (self.data_length // self.batch_size) - 2
		self.worker_pool = ThreadPool(processes=num_of_workers)

	def make_batch(self, data):
		if data is not None:
			self.batches.append(np.array([cv.cvtColor(cv.imread(im_p), cv.COLOR_BGR2RGB) / 127.5 - 1.0 for im_p in data]).astype(np.float32))

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

	def get_batch(self) -> np.ndarray:
		while not self.batches: time.sleep(0.01)
		return self.batches.popleft()

	def get_larger_batch(self, num_of_batches_to_merge:int):
		batch = []
		for _ in range(num_of_batches_to_merge):
			while not self.batches: time.sleep(0.01)
			for img in self.batches.popleft():
				batch.append(img)
		return np.array(batch)