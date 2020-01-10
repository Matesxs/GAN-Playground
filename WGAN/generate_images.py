from keras.models import Model
from keras.layers import Input, Dense
import os
import numpy as np
import cv2 as cv

from modules import generator_models_spreadsheet, critic_models_spreadsheet

latent_dim = 100
img_shape = (64, 64, 3)
generator_model_name = "mod_min_3upscl"
discriminator_model_name = "mod_min_5layers"
generator_weights_path = None
discriminator_weights_path = None

num_of_images = 1000
image_save_path = "generated_images"

lat_input = Input(shape=(latent_dim,))
preq_gen = getattr(generator_models_spreadsheet, generator_model_name)(lat_input, img_shape, img_shape[2])
gen_mod = Model(lat_input, preq_gen, name="generator")
if generator_weights_path: gen_mod.load_weights(generator_weights_path)

im_input = Input(shape=img_shape)
preq_disc = getattr(critic_models_spreadsheet, discriminator_model_name)(im_input, )
preq_disc = Dense(1, activation="sigmoid")(preq_disc)
disc_mod = Model(im_input, preq_disc)
if discriminator_weights_path: disc_mod.load_weights(discriminator_weights_path)

noise = np.random.normal(np.random.normal(0.0, 1.0, size=(num_of_images, latent_dim)))
gen_images = gen_mod.predict(noise)
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
	print(disc_mod.predict(image))
	cv.imwrite(f"{image_save_path}/img_{idx+1}.png", image)