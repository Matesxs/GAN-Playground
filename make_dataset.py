import os
import cv2 as cv

scaled_dim = (64, 64)
input_folder = "dataset/cats"
output_folder = "dataset/normalized_cats"

raw_file_names = os.listdir(input_folder)
if not os.path.isdir(output_folder):
	os.mkdir(output_folder)

for idx, file_name in enumerate(raw_file_names):
	if os.path.isfile(f"{input_folder}/" + file_name):
		image = cv.imread(f"{input_folder}/" + file_name)
		image = cv.resize(image, (scaled_dim[0], scaled_dim[1]), interpolation=cv.INTER_CUBIC)
		cv.imwrite(f"{output_folder}/{idx}.png", image)