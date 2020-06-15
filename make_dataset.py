import os
import cv2 as cv
import hashlib
from multiprocessing.pool import ThreadPool

worker_pool = ThreadPool(processes=8)

scaled_dim = (64, 64)
input_folder = "dataset/faces"
output_folder = "dataset/normalized_faces"

raw_file_names = os.listdir(input_folder)
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# Detect and remove duplicates
duplicates = []
used_hashes = []
def check_for_duplicates(filename):
	file_path = os.path.join(input_folder, filename)
	if os.path.isfile(file_path):
		with open(file_path, 'rb') as f:
			filehash = hashlib.md5(f.read()).hexdigest()

			if filehash not in used_hashes:
				used_hashes.append(filehash)
			else:
				duplicates.append(file_path)

worker_pool.map(check_for_duplicates, raw_file_names)

print(f"Found {len(duplicates)} duplicates")
def remove_duplicate(file_path):
	try:
		os.remove(file_path)
	except:
		pass

worker_pool.map(remove_duplicate, duplicates)

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