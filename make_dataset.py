import os
import cv2 as cv

scaled_size = 64
folder = "training_data"

raw_file_names = os.listdir(folder)
if not os.path.isdir(f"{folder}/normalized"):
	os.mkdir(f"{folder}/normalized")

for file_name in raw_file_names:
	if os.path.isfile(f"{folder}/" + file_name):
		image = cv.imread(f"{folder}/" + file_name)
		image = cv.resize(image, (scaled_size, scaled_size), interpolation=cv.INTER_CUBIC)
		cv.imwrite(f"{folder}/normalized/" + file_name, image)