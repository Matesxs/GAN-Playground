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
		self.save_path = save_path

		self.terminate = False

	def run(self) -> None:
		while not self.terminate or self.data:
			if len(self.data) > 0:
				with open(f"{self.save_path}/training_stats.csv", "a+") as f:
					writer = csv.writer(f)
					for _ in range(len(self.data)):
						writer.writerow(self.data.popleft())
					f.close()
			time.sleep(0.04)

	def apptend_stats(self, stats:list):
		self.data.append(stats)