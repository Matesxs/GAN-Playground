from keras.models import Model
from keras.layers import Input
import os
import numpy as np
import cv2 as cv

from modules import generator_models_spreadsheet

latent_dim = 512
img_shape = (64, 64, 3)
model_name = "mod_ext_4upscl"
model_weights_path = "D:/Projekty/Python/GANTest/trained_weights/100/generator_mod_ext_4upscl.h5"

num_of_images = 300
image_save_path = "generated_images"

inp = Input(shape=(latent_dim,))
m = getattr(generator_models_spreadsheet, model_name)(inp, img_shape, img_shape[2])
model = Model(inp, m, name="generator")
if model_weights_path: model.load_weights(model_weights_path)

noise = np.random.normal(np.random.normal(0.0, 1.0, size=(num_of_images, latent_dim)))
gen_images = model.predict(noise)
gen_images = (0.5 * gen_images + 0.5) * 255

if not os.path.isdir(image_save_path):
	os.makedirs(image_save_path)
else:
	files_in_out_folder = os.listdir(image_save_path)
	for file_name in files_in_out_folder:
		path = os.path.join(image_save_path, file_name)
		if os.path.isfile(path):
			os.remove(path)

for idx, image in enumerate(gen_images):
	image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
	cv.imwrite(f"{image_save_path}/img_{idx+1}.png", image)