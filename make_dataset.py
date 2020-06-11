import os
import cv2 as cv
import hashlib

scaled_dim = (64, 64)
input_folder = "dataset/dogs"
output_folder = "dataset/normalized_dogs"

raw_file_names = os.listdir(input_folder)
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# Detect and remove duplicates
duplicates = []
used_hashes = []
for index, filename in enumerate(raw_file_names):
	file_path = os.path.join(input_folder, filename)
	if os.path.isfile(file_path):
		with open(file_path, 'rb') as f:
			filehash = hashlib.md5(f.read()).hexdigest()

			if filehash not in used_hashes:
				used_hashes.append(filehash)
			else:
				duplicates.append(file_path)

print(f"Found {len(duplicates)} duplicates")
for file_path in duplicates:
	try:
		os.remove(file_path)
	except:
		pass

for idx, file_name in enumerate(raw_file_names):
	file_path = os.path.join(input_folder, file_name)
	if os.path.isfile(file_path):
		image = cv.imread(file_path)
		if image is not None:
			image = cv.resize(image, (scaled_dim[0], scaled_dim[1]), interpolation=cv.INTER_CUBIC)
			cv.imwrite(f"{output_folder}/{idx}.png", image)