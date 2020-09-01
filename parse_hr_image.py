import os
import cv2 as cv

from modules.helpers import get_paths_of_files_from_path

HR_IMAGES_FOLDER = r"C:\Users\marti\OneDrive\Pictures"
OUTPUT_FOLDER = "test_folder"
TARGET_SHAPE = (256, 256)

assert os.path.exists(HR_IMAGES_FOLDER), "Input folder doesnt exist"
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

hr_image_paths = get_paths_of_files_from_path(HR_IMAGES_FOLDER)
assert len(hr_image_paths) > 0, "High resulution image folder is empty"

for hr_image_path in hr_image_paths:
  if not os.path.exists(hr_image_path): continue

  hr_image = cv.imread(hr_image_path)
  if hr_image is None: continue

  print(hr_image.shape[:-1])