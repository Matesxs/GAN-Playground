import os
import cv2 as cv
import hashlib
import shutil
from multiprocessing.pool import ThreadPool

datasets_folder = "datasets"

assert os.path.exists(datasets_folder) and os.path.isdir(datasets_folder), "Invalid datasets folder"

dataset_list = [x for x in os.listdir(datasets_folder) if os.path.isdir(os.path.join(datasets_folder, x)) and "normalized" not in x]
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
		input_folder = os.path.join(dataset_folder, selected_dataset_name)
		output_folder = os.path.join(dataset_folder, f"{selected_dataset_name}_normalized__{selected_x_dimension}x{selected_y_dimension}")
		scaled_dim = (selected_x_dimension, selected_y_dimension)
		print(f"Dataset {selected_dataset_name} was selected with target dimensions: {scaled_dim}")
		break
	except:
		print("")
		continue

assert input_folder is not None and output_folder is not None and scaled_dim is not None, "Invalid settings"
assert os.path.exists(input_folder), "Input folder doesnt exist"

worker_pool = ThreadPool(processes=16)

raw_file_names = os.listdir(input_folder)
if os.path.exists(output_folder): shutil.rmtree(output_folder)
os.mkdir(output_folder)

print(f"Found {len(raw_file_names)} files")

# Detect and remove duplicates
duplicate_files = []
used_hashes = []
def check_for_duplicates(filename):
	file_path = os.path.join(input_folder, filename)
	if os.path.isfile(file_path):
		with open(file_path, 'rb') as f:
			filehash = hashlib.md5(f.read()).hexdigest()

			if filehash not in used_hashes:
				used_hashes.append(filehash)
			else:
				duplicate_files.append(file_path)

worker_pool.map(check_for_duplicates, raw_file_names)

print(f"Found {len(duplicate_files)} duplicates")
def remove_duplicate(file_path):
	try:
		os.remove(file_path)
	except:
		pass

worker_pool.map(remove_duplicate, duplicate_files)

print(f"{len(raw_file_names) - len(duplicate_files)} files to normalize")

def resize_and_save_file(args):
	file_path = os.path.join(input_folder, args[1])
	if os.path.exists(file_path) and os.path.isfile(file_path):
		try:
			image = cv.imread(file_path)
			if image is not None:
				image = cv.resize(image, (scaled_dim[0], scaled_dim[1]), interpolation=cv.INTER_CUBIC)
				cv.imwrite(f"{output_folder}/{args[0]}.png", image)
		except:
			pass

worker_pool.map(resize_and_save_file, enumerate(raw_file_names))
worker_pool.close()