from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import time
import shutil
import os
import asyncio
from tqdm import tqdm
import concurrent.futures
import hashlib
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

IMGS_TO_DOWNLOAD = 10_000
SAVE_PATH = "datasets/random_images"
PATH_TO_CHROME_DRIVER = r"lib/chromedriver.exe"
SCRAPE_URL = "https://source.unsplash.com/random/1024x1024" # https://picsum.photos/1024 https://loremflickr.com/1024/1024/all
DELAY_AFTER_GETTING_URL = 5

NUM_OF_DOWNLOAD_WORKERS = 1

driver_opt = Options()
driver_opt.add_argument("--headless")

def download_image(img_url):
  img_stream = requests.get(img_url, stream=True)
  img_stream.raw.decode_content = True

  path = os.path.join(SAVE_PATH, f"{str(time.time()).replace('.', '_')}.png")
  while os.path.exists(path):
    time.sleep(0.001)
    path = os.path.join(SAVE_PATH, f"{str(time.time()).replace('.', '_')}.png")

  local_file = open(path, "wb")
  shutil.copyfileobj(img_stream.raw, local_file)

  del img_stream
  local_file.close()

image_urls = []
def worker(images_to_download):
  driver = webdriver.Chrome(executable_path=PATH_TO_CHROME_DRIVER, service_log_path=os.devnull, options=driver_opt)

  for _ in range(images_to_download) if NUM_OF_DOWNLOAD_WORKERS > 1 else tqdm(range(images_to_download)):
    try:
      driver.get(SCRAPE_URL)
      el = driver.find_element_by_xpath("/html/body/img")
      if el.get_attribute("src") not in image_urls:
        image_urls.append(el.get_attribute("src"))
      time.sleep(DELAY_AFTER_GETTING_URL)
    except NoSuchElementException:
      pass

  time.sleep(1)
  driver.close()

async def scraper_manager():
  images_per_worker = IMGS_TO_DOWNLOAD // NUM_OF_DOWNLOAD_WORKERS
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_OF_DOWNLOAD_WORKERS)

  futures = []
  for _ in range(NUM_OF_DOWNLOAD_WORKERS):
    future = executor.submit(worker, images_to_download=images_per_worker)
    futures.append(asyncio.wrap_future(future))

  for future in futures:
    await asyncio.wait_for(future, timeout=None, loop=asyncio.get_event_loop())
  executor.shutdown()

used_hashes = []
duplicate_file_paths = []
def check_for_duplicates(file_path):
  if os.path.isfile(file_path):
    with open(file_path, 'rb') as f:
      filehash = hashlib.md5(f.read()).hexdigest()

      if filehash not in used_hashes:
        used_hashes.append(filehash)
      else:
        duplicate_file_paths.append(file_path)

def remove_duplicate(file_path):
  try:
    os.remove(file_path)
  except:
    pass

if __name__ == '__main__':
  try:
    asyncio.run(scraper_manager())
  except KeyboardInterrupt:
    pass
  except Exception:
    pass

  print(f"Found {len(image_urls)} images")

  with Pool(processes=cpu_count()) as p:
    p.map(download_image, image_urls)

  all_images_paths = [os.path.join(SAVE_PATH, x) for x in os.listdir(SAVE_PATH)]
  with ThreadPool(processes=8) as tp:
    tp.map(check_for_duplicates, all_images_paths)
    print(f"Found {len(duplicate_file_paths)} duplicates")
    if len(duplicate_file_paths) > 0:
      tp.map(remove_duplicate, duplicate_file_paths)

  all_images_paths = [os.path.join(SAVE_PATH, x) for x in os.listdir(SAVE_PATH)]
  print(f"In {SAVE_PATH} folder is {len(all_images_paths)} images")
  print(f"Added {len(image_urls) - len(duplicate_file_paths)} images")

  if os.path.exists("debug.log"): os.remove("debug.log")