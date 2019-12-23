import os
import cv2 as cv

scaled_size = 64

raw_file_names = os.listdir("data")
if not os.path.isdir("data/normalized"):
	os.mkdir("data/normalized")

for file_name in raw_file_names:
	if os.path.isfile("data/" + file_name):
		image = cv.imread("data/" + file_name)
		image = cv.resize(image, (scaled_size, scaled_size), interpolation=cv.INTER_CUBIC)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		cv.imwrite("data/normalized/" + file_name, image)