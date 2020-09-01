import os
import cv2 as cv
import hashlib
import shutil
import ntpath
import random
import numpy as np
import logging
from multiprocessing.pool import ThreadPool

from modules.helpers import get_paths_of_files_from_path

logging.getLogger("opencv-python").setLevel(logging.CRITICAL)

datasets_folder = "datasets"

assert os.path.exists(datasets_folder) and os.path.isdir(datasets_folder), "Invalid datasets folder"

dataset_list = [x for x in os.listdir(datasets_folder) if os.path.isdir(os.path.join(datasets_folder, x)) and "normalized" not in x]
dataset_list.append("All (All datasets merged)")

selected_dataset_name = None
selected_x_dimension = None
selected_y_dimension = None
output_folder = None
input_folder = None
scaled_dim = None
testing_split = None

crop_images = False
ignore_smaller_images_than_target = False
deep_check_duplicates = False

while True:
  print("Avaible input datasets:")
  for i, dataset_name in enumerate(dataset_list):
    print(f"{i} - {dataset_name}")

  try:
    selected_dataset_index = int(input("Selected datasets: "))
    if selected_dataset_index >= len(dataset_list):
      print("")
      continue

    selected_x_dimension = None
    selected_y_dimension = None
    while True:
      try:
        selected_x_dimension = int(input("Target x dimension: "))
        selected_y_dimension = int(input("Target y dimension: "))
        break
      except:
        continue

    ignore_smaller_images_than_target = (input("Ignore smaller images than target (y/n): ").lower() == "y")
    crop_images = (input("Crop input images to target aspect ratio (y/n): ").lower() == "y")
    deep_check_duplicates = (input("Check deeply for duplicates (y/n): ").lower() == "y")

    try:
      testing_split = float(input("Testing split (0 - 1), leave blank for not plitting: "))
      if testing_split > 1: testing_split = None
    except:
      pass

    selected_dataset_name = dataset_list[selected_dataset_index]

    if selected_dataset_name == "All (All datasets merged)":
      input_folder = [os.path.join(datasets_folder, x) for x in dataset_list if x != "All (All datasets merged)"]
      selected_dataset_name = "all"
    else:
      input_folder = os.path.join(datasets_folder, selected_dataset_name)

    output_folder = os.path.join(datasets_folder, f"{selected_dataset_name}_normalized__{selected_x_dimension}x{selected_y_dimension}" + ("__train" if testing_split else ""))
    scaled_dim = (selected_x_dimension, selected_y_dimension)

    print(f"Dataset {selected_dataset_name} was selected with target dimensions: {scaled_dim}" + (f" with test split {testing_split}" if testing_split else ""))
    break
  except:
    print("")
    continue

assert input_folder is not None and output_folder is not None and scaled_dim is not None, "Invalid settings"
if isinstance(input_folder, str):
  assert os.path.exists(input_folder), "Input folder doesnt exist"
  raw_file_paths = get_paths_of_files_from_path(input_folder)
elif isinstance(input_folder, list):
  raw_file_paths = []
  for y in input_folder:
    for x in os.listdir(y):
      raw_file_paths.append(os.path.join(y, x))
else:
  raise Exception("Invalid input folder format")

worker_pool = ThreadPool(processes=16)

if os.path.exists(output_folder): shutil.rmtree(output_folder)
os.mkdir(output_folder)

print(f"Found {len(raw_file_paths)} files")

# Detect duplicates
duplicate_files = []
used_hashes = []
filepaths_to_use = []

def check_for_duplicates(file_path):
  if os.path.isfile(file_path):
    with open(file_path, 'rb') as f:
      filehash = hashlib.md5(f.read()).hexdigest()

      if filehash not in used_hashes:
        used_hashes.append(filehash)
        filepaths_to_use.append(file_path)
      else:
        duplicate_files.append(file_path)

worker_pool.map(check_for_duplicates, raw_file_paths)

def deeply_check_for_duplicates():
  global filepaths_to_use

  def test_duplicity(test_path):
    if test_path not in duplicate_files:
      test_image = cv.imread(test_path)

      if np.array_equal(main_image, test_image):
        duplicate_files.append(test_path)
        if test_path not in paths_to_remove:
          paths_to_remove.append(test_path)

  copy_of_filepaths_to_use = filepaths_to_use.copy()
  paths_to_remove = []

  for idx, filepath in enumerate(copy_of_filepaths_to_use):
    if (idx + 1) >= len(copy_of_filepaths_to_use): break
    if filepath not in duplicate_files:

      main_image = cv.imread(filepath)
      worker_pool.map(test_duplicity, copy_of_filepaths_to_use[idx + 1:])


  filepaths_to_use = [path for path in files_to_move_paths if path not in paths_to_remove]

# Make array to array comparison of images
if deep_check_duplicates:
  worker_pool.apply(deeply_check_for_duplicates)

print(f"Found {len(duplicate_files)} duplicates")
def remove_duplicate(file_path):
  try:
    os.remove(file_path)
  except:
    pass

if isinstance(input_folder, str):
  worker_pool.map(remove_duplicate, duplicate_files)

print(f"{len(filepaths_to_use)} files to normalize")

target_aspect_ratio = scaled_dim[0]/scaled_dim[1]
def crop_image(image, current_aspect_ratio, current_shape):
  if target_aspect_ratio < current_aspect_ratio:
    new_width = current_shape[0] / target_aspect_ratio
    width_diff = current_shape[1] - new_width
    image = image[:, int(width_diff // 2):int(current_shape[1] - (width_diff // 2)), :]
  else:
    new_height = current_shape[1] / target_aspect_ratio
    height_diff = current_shape[0] - new_height
    image = image[int(height_diff // 2):int(current_shape[0] - (height_diff // 2)), :, :]

  return image

ignored_images = 0
output_to_original_filepath = {}
def resize_and_save_file(args):
  global ignored_images

  if os.path.exists(args[1]) and os.path.isfile(args[1]):
    # try:
      image = cv.imread(args[1])

      if image is not None:
        if crop_images:
          orig_shape = image.shape[:-1]
          original_aspect_ratio = orig_shape[1]/orig_shape[0]
          if target_aspect_ratio != original_aspect_ratio:
            image = crop_image(image, original_aspect_ratio, orig_shape)

        orig_shape = image.shape[:-1]

        if ignore_smaller_images_than_target:
          if orig_shape[0] < scaled_dim[1] or orig_shape[1] < scaled_dim[0]:
            ignored_images += 1
            return

        if orig_shape[0] != scaled_dim[1] or orig_shape[1] != scaled_dim[0]:
          interpolation = cv.INTER_AREA
          if orig_shape[0] <= scaled_dim[1] or orig_shape[1] <= scaled_dim[0]:
            interpolation = cv.INTER_CUBIC

          image = cv.resize(image, (scaled_dim[0], scaled_dim[1]), interpolation=interpolation)

        cv.imwrite(f"{output_folder}/{args[0]}.png", image)
        output_to_original_filepath[f"{output_folder}/{args[0]}.png"] = args[1]
    # except Exception as e:
    #   try:
    #     os.remove(f"{output_folder}/{args[0]}.png")
    #   except:
    #     pass

worker_pool.map(resize_and_save_file, enumerate(filepaths_to_use))
if ignore_smaller_images_than_target:
  print(f"Ignored {ignored_images} due to low resolution")

# Detect duplicates
used_hashes = []
resized_duplicates = 0
def delete_resized_duplicates(file_path):
  global resized_duplicates

  if os.path.isfile(file_path):
    with open(file_path, 'rb') as f:
      filehash = hashlib.md5(f.read()).hexdigest()

      if filehash not in used_hashes:
        used_hashes.append(filehash)
      else:
        try:
          os.remove(file_path)
          if isinstance(input_folder, str):
            os.remove(output_to_original_filepath[file_path])
          resized_duplicates += 1
        except:
          pass

output_files = get_paths_of_files_from_path(output_folder)
worker_pool.map(delete_resized_duplicates, output_files)
print(f"Deleted {resized_duplicates} already resized duplicates")

if testing_split:
  testing_folder_path = os.path.join(datasets_folder, f"{selected_dataset_name}_normalized__{selected_x_dimension}x{selected_y_dimension}__test")
  if os.path.exists(testing_folder_path): shutil.rmtree(testing_folder_path)
  os.mkdir(testing_folder_path)

  train_folder_files = get_paths_of_files_from_path(output_folder)
  num_test_files_count = int(len(train_folder_files) * testing_split)

  random.shuffle(train_folder_files)
  files_to_move_paths = train_folder_files[:num_test_files_count]

  def move_file(original_path):
    file_name = ntpath.basename(original_path)
    shutil.move(original_path, os.path.join(testing_folder_path, file_name))

  print(f"{num_test_files_count} files will be moved to test folder")
  worker_pool.map(move_file, files_to_move_paths)

worker_pool.close()