from threading import Thread
from collections import deque
import os
import csv
import time

class StatSaver(Thread):
	def __init__(self, save_path:str) -> None:
		super().__init__()
		self.daemon = True

		self.data = deque()

		if not os.path.exists(save_path): os.makedirs(save_path)
		self.save_file = open(f"{save_path}/training_stats.csv", "a+")
		self.writer = csv.writer(self.save_file)

		self.terminate = False

	def run(self) -> None:
		while not self.terminate or self.data:
			if self.data:
				for stat in self.data:
					self.writer.writerow(stat)
			time.sleep(0.02)
		self.save_file.close()

	def apptend_stats(self, stats:list):
		self.data.append(stats)