import os
import cv2 as cv
import random
import time

from gans.utils.helpers import walk_path

DATASET_FOLDER = "datasets"

datasets = os.listdir(DATASET_FOLDER)
datasets = [p for p in datasets if os.path.isdir(os.path.join(DATASET_FOLDER, p))]

print("Datasets:")
for dataset_idx, dataset in enumerate(datasets):
  print(f"{dataset_idx}: {dataset}")

selected_dataset = None
while selected_dataset is None:
  try:
    sel_dataset_index = int(input("Select dataset: "))
    selected_dataset = datasets[sel_dataset_index]
  except:
    print("Invalid dataset selected")

image_paths = walk_path(os.path.join(DATASET_FOLDER, selected_dataset))
random.shuffle(image_paths)

number_of_images = len(image_paths)
for img_idx, image_path in enumerate(image_paths):
  try:
    image = cv.imread(image_path)
    image = cv.resize(image, (512, 512), interpolation=cv.INTER_CUBIC)

    cv.imshow("Image", image)
    cv.setWindowTitle("Image", f"{img_idx + 1}/{number_of_images}")
    print(f"Delete {image_path}? (enter = yes, others = no)")
    key = cv.waitKey(0)

    if int(key) == 13:
      print(f"Deleting {image_path}")
      os.remove(image_path)

    time.sleep(0.1)
  except KeyboardInterrupt:
    print("Exiting")
