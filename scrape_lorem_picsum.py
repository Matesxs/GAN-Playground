from selenium import webdriver
import requests
import time
import shutil
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

IMGS_TO_DOWNLOAD = 50_000
SAVE_PATH = "datasets/random_images"
PATH_TO_CHROME_DRIVER = r"lib/chromedriver.exe"
RESOLUTION = 1024

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

if __name__ == '__main__':
  driver = webdriver.Chrome(executable_path=PATH_TO_CHROME_DRIVER)

  image_urls = []
  for _ in tqdm(range(IMGS_TO_DOWNLOAD)):
    try:
      driver.get(f"https://picsum.photos/{RESOLUTION}")
      el = driver.find_element_by_xpath("/html/body/img")
      if el.get_attribute("src") not in image_urls:
        image_urls.append(el.get_attribute("src"))
      time.sleep(0.01)
    except KeyboardInterrupt:
      break

  print(f"Found {len(image_urls)} unique images")
  time.sleep(1)
  driver.close()

  with Pool(processes=cpu_count()) as p:
    p.map(download_image, image_urls)