import os
import cv2 as cv

scaled_dim = (64, 64)
folder = "../dataset/dogs"

raw_file_names = os.listdir(folder)
if not os.path.isdir(f"{folder}/normalized"):
	os.mkdir(f"{folder}/normalized")

for idx, file_name in enumerate(raw_file_names):
	if os.path.isfile(f"{folder}/" + file_name):
		image = cv.imread(f"{folder}/" + file_name)
		image = cv.resize(image, (scaled_dim[0], scaled_dim[1]), interpolation=cv.INTER_CUBIC)
		cv.imwrite(f"{folder}/normalized/{idx}.png", image)