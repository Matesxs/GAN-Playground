try:
  from urlparse import urljoin
except ImportError:
  from urllib.parse import urljoin

import os
import requests
from bs4 import BeautifulSoup
from threading import Thread
import time
from collections import deque

# Script for scraping generators
# Format: (url, save_path, num_of_images_to_keep, is_straight_image)
PARAMS = [
  ("https://thispersondoesnotexist.com", "datasets/unfiltered_faces", 20_000, False, 30_000),
  ("https://thiscatdoesnotexist.com", "datasets/unfiltered_cats", 20_000, True, 50_000)
]

class Scraper:
  def __init__(self, save_folder_path: str, start_index: int = 0):
    self.save_folder_path = save_folder_path
    self.start_index = start_index
    self.session = requests.Session()
    self.session.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36 OPR/66.0.3515.36"}

    requests.packages.urllib3.disable_warnings()  # turn off SSL warnings

  def visit_url(self, url):
    content = self.session.get(url, verify=False).content
    soup = BeautifulSoup(content, "lxml")

    for img in soup.select("img[src]"):
      image_url = img["src"]
      if not image_url.startswith(("data:image", "javascript")):
        if "http://" in image_url:
          self.download_image(image_url)
        else:
          self.download_image(urljoin(url, image_url))

  def find_free_name(self):
    for i in range(self.start_index, len(os.listdir(self.save_folder_path)) + 1):
      name = f"{self.save_folder_path}/{i}.jpg"
      if not os.path.exists(name): return name
    raise Exception("Cant find replacement name")

  def download_image(self, image_url):
    local_filename = f"{self.save_folder_path}/{len(os.listdir(self.save_folder_path)) + self.start_index}.jpg"
    if os.path.exists(local_filename):
      local_filename = self.find_free_name()

    r = self.session.get(image_url, stream=True, verify=False)
    with open(local_filename, 'wb') as f:
      for chunk in r.iter_content(chunk_size=1024):
        f.write(chunk)


class Downloader(Thread):
  def __init__(self, idx: int, repeats: int, save_path: str, url: str, force: bool, start_index: int = 0):
    super().__init__()
    self.daemon = True

    self.force = force
    self.idx = idx
    self.num_of_repeats = repeats
    self.save_path = save_path
    self.url = url

    if not os.path.exists(self.save_path): os.makedirs(self.save_path)
    self.scraper = Scraper(self.save_path, start_index)

  def run(self):
    print(f"Downloader {self.idx} started - {self.url}")
    while len(os.listdir(self.save_path)) < self.num_of_repeats:
      try:
        if self.force:
          self.scraper.download_image(self.url)
        else:
          self.scraper.visit_url(self.url)
        time.sleep(2)
      except Exception:
        time.sleep(5)
        self.scraper = Scraper(self.save_path)
    print(f"Downloader {self.idx} finished - {self.url}")


if __name__ == '__main__':
  workers = deque()
  for url, save_path, num_of_images, force, start_index in PARAMS:
    idx = len(workers)
    worker = Downloader(idx, num_of_images, save_path, url, force, start_index)
    worker.start()
    workers.append(worker)

  for w in workers:
    w.join()
