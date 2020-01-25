try:
	from urlparse import urljoin
except ImportError:
	from urllib.parse import urljoin

import os
import requests
from bs4 import BeautifulSoup
from threading import Thread
import time

# https://thiscatdoesnotexist.com
URL = "https://thispersondoesnotexist.com"
SAVE_PATH = "../dataset/faces"
NUM_OF_IMAGES = 20_000

class Scraper:
	def __init__(self, save_folder_path:str):
		self.save_folder_path = save_folder_path
		self.session = requests.Session()
		self.session.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.109 Safari/537.36"}

		requests.packages.urllib3.disable_warnings()  # turn off SSL warnings

	def visit_url(self, url):
		content = self.session.get(url, verify=False).content
		soup = BeautifulSoup(content, "lxml")

		for img in soup.select("img[src]"):
			image_url = img["src"]
			if not image_url.startswith(("data:image", "javascript")):
				self.download_image(urljoin(url, image_url))

	def download_image(self, image_url):
		local_filename = f"{self.save_folder_path}/{len(os.listdir(self.save_folder_path))}.jpg"

		r = self.session.get(image_url, stream=True, verify=False)
		with open(local_filename, 'wb') as f:
			for chunk in r.iter_content(chunk_size=1024):
				f.write(chunk)

class Downloader(Thread):
	def __init__(self, repeats:int):
		super().__init__()
		self.daemon = True

		self.num_of_repeats = repeats
		self.scraper = Scraper(SAVE_PATH)

	def run(self):
		for _ in range(self.num_of_repeats):
			self.scraper.visit_url(URL)
			time.sleep(2)

if __name__ == '__main__':
	worker = Downloader(NUM_OF_IMAGES)
	worker.start()
	worker.join()