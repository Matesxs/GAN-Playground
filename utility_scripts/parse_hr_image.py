import os
import cv2 as cv

from ..modules.utils.helpers import get_paths_of_files_from_path

HR_IMAGES_FOLDER = r""
OUTPUT_FOLDER = r""
TARGET_SHAPE = (256, 256)

assert os.path.exists(HR_IMAGES_FOLDER), "Input folder doesnt exist"
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

hr_image_paths = get_paths_of_files_from_path(HR_IMAGES_FOLDER, only_files=True)
assert len(hr_image_paths) > 0, "High resulution image folder is empty"

for idx, hr_image_path in enumerate(hr_image_paths):
  if not os.path.exists(hr_image_path):
    continue

  hr_image = cv.imread(hr_image_path)
  if hr_image is None:
    continue

  iter_over_y = int(hr_image.shape[0] / TARGET_SHAPE[1])
  iter_over_x = int(hr_image.shape[1] / TARGET_SHAPE[0])

  if iter_over_x == 0 and iter_over_y == 0:
    continue
  
  for y_idx in range(iter_over_y):
    for x_idx in range(iter_over_x):
      image_part = hr_image[y_idx * TARGET_SHAPE[1]:(y_idx + 1) * TARGET_SHAPE[1], x_idx * TARGET_SHAPE[0]:(x_idx + 1) * TARGET_SHAPE[0], :]
      cv.imwrite(os.path.join(OUTPUT_FOLDER, f"{idx}_{(y_idx * TARGET_SHAPE[1]) + x_idx}.png"), image_part)