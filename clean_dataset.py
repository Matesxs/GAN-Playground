import os
import imagesize
from multiprocessing.pool import ThreadPool

from gans.utils.helpers import walk_path

DATASETS_FOLDER_PATH = r"datasets"

assert os.path.exists(DATASETS_FOLDER_PATH) and os.path.isdir(DATASETS_FOLDER_PATH), "Invalid datasets folder"

dataset_list = [x for x in os.listdir(DATASETS_FOLDER_PATH) if os.path.isdir(os.path.join(DATASETS_FOLDER_PATH, x))]

selected_dataset_name = None
selected_xdim = None
selected_ydim = None

while True:
  print("Avaible input datasets:")
  for i, dataset_name in enumerate(dataset_list):
    print(f"{i} - {dataset_name}")

  selected_dataset_index = int(input("Selected datasets: "))
  if selected_dataset_index >= len(dataset_list):
    print("")
    continue

  selected_dataset_name = dataset_list[selected_dataset_index]
  break

while True:
  try:
    selected_xdim = int(input("Target x dimension: "))
    selected_ydim = int(input("Target y dimension: "))
    break
  except:
    print("")
    continue

dataset_path = os.path.join(DATASETS_FOLDER_PATH, selected_dataset_name)
image_list = walk_path(dataset_path)
print(f"Found {len(image_list)} images")

deleted_images = 0
def check_image(image_path):
  global deleted_images
  shape = imagesize.get(image_path)

  if shape[0] < selected_xdim or shape[1] < selected_ydim:
    os.remove(image_path)
    deleted_images += 1

try:
  with ThreadPool(16) as pool:
    pool.map(check_image, image_list)
except KeyboardInterrupt:
  pass

print(f"Deleted {deleted_images} images")