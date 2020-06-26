import os
import cv2 as cv
import hashlib
import shutil
from multiprocessing.pool import ThreadPool

datasets_folder = "datasets"

assert os.path.exists(datasets_folder) and os.path.isdir(datasets_folder), "Invalid datasets folder"

dataset_list = [x for x in os.listdir(datasets_folder) if os.path.isdir(os.path.join(datasets_folder, x)) and "normalized" not in x]
dataset_list.append("All (All datasets merged)")
output_folder = None
input_folder = None
scaled_dim = None
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

    selected_dataset_name = dataset_list[selected_dataset_index]
    if selected_dataset_name == "All (All datasets merged)":
      input_folder = [os.path.join(datasets_folder, x) for x in dataset_list if x != "All (All datasets merged)"]
      selected_dataset_name = "all"
    else:
      input_folder = os.path.join(datasets_folder, selected_dataset_name)
    output_folder = os.path.join(datasets_folder, f"{selected_dataset_name}_normalized__{selected_x_dimension}x{selected_y_dimension}")
    scaled_dim = (selected_x_dimension, selected_y_dimension)
    print(f"Dataset {selected_dataset_name} was selected with target dimensions: {scaled_dim}")
    break
  except:
    print("")
    continue

assert input_folder is not None and output_folder is not None and scaled_dim is not None, "Invalid settings"
if isinstance(input_folder, str):
  assert os.path.exists(input_folder), "Input folder doesnt exist"
  raw_file_paths = [os.path.join(input_folder, x) for x in os.listdir(input_folder)]
elif isinstance(input_folder, list):
  raw_file_paths = []
  for y in input_folder:
    for x in os.listdir(y):
      raw_file_paths.append(os.path.join(y, x))
else:
  raise Exception("Invalid input folder format")

worker_pool = ThreadPool(processes=8)

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

print(f"Found {len(duplicate_files)} duplicates")
def remove_duplicate(file_path):
  try:
    os.remove(file_path)
  except:
    pass

if isinstance(input_folder, str):
  worker_pool.map(remove_duplicate, duplicate_files)

print(f"{len(filepaths_to_use)} files to normalize")

def resize_and_save_file(args):
  if os.path.exists(args[1]) and os.path.isfile(args[1]):
    try:
      image = cv.imread(args[1])
      if image is not None:
        orig_shape = image.shape[:-1]
        interpolation = cv.INTER_AREA
        if orig_shape[0] <= scaled_dim[0] or orig_shape[1] <= scaled_dim[1]:
          interpolation = cv.INTER_CUBIC

        image = cv.resize(image, (scaled_dim[0], scaled_dim[1]), interpolation=interpolation)
        cv.imwrite(f"{output_folder}/{args[0]}.png", image)
    except:
      try:
        os.remove(f"{output_folder}/{args[0]}.png")
      except:
        pass

worker_pool.map(resize_and_save_file, enumerate(filepaths_to_use))
worker_pool.close()