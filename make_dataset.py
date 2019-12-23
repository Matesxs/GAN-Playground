import os
import cv2 as cv

scaled_size = 64

raw_file_names = os.listdir("training_data")
if not os.path.isdir("training_data/normalized"):
	os.mkdir("training_data/normalized")

for file_name in raw_file_names:
	if os.path.isfile("training_data/" + file_name):
		image = cv.imread("training_data/" + file_name)
		image = cv.resize(image, (scaled_size, scaled_size), interpolation=cv.INTER_CUBIC)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		cv.imwrite("training_data/normalized/" + file_name, image)