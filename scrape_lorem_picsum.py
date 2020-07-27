from selenium import webdriver
import requests
import time
import shutil
import os
import asyncio
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Pool, cpu_count

IMGS_TO_DOWNLOAD = 50_000
SAVE_PATH = "datasets/random_images"
PATH_TO_CHROME_DRIVER = r"lib/chromedriver.exe"
SCRAPE_URL = "https://source.unsplash.com/random/1024x1024" # https://picsum.photos/1024 https://loremflickr.com/1024/1024/all
DELAY_AFTER_GETTING_URL = 5

NUM_OF_DOWNLOAD_WORKERS = 1

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
  driver = webdriver.Chrome(executable_path=PATH_TO_CHROME_DRIVER)

  for _ in range(images_to_download) if NUM_OF_DOWNLOAD_WORKERS > 1 else tqdm(range(images_to_download)):
    try:
      driver.get(SCRAPE_URL)
      el = driver.find_element_by_xpath("/html/body/img")
      if el.get_attribute("src") not in image_urls:
        image_urls.append(el.get_attribute("src"))
      time.sleep(DELAY_AFTER_GETTING_URL)
    except KeyboardInterrupt:
      break
    except Exception:
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

if __name__ == '__main__':
  try:
    asyncio.run(scraper_manager())
  except KeyboardInterrupt:
    pass
  except Exception:
    pass

  print(f"Found {len(image_urls)} unique images")

  with Pool(processes=cpu_count()) as p:
    p.map(download_image, image_urls)