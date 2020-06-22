import os
import sys
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdin = stdin
sys.stderr = stderr

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
			tf.config.experimental.set_memory_growth(gpus[0], True)
	except:
		pass

from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import cv2 as cv

from modules.models import discriminator_models_spreadsheet, generator_models_spreadsheet

latent_dim = 128
img_shape = (64, 64, 3)
final_size = (256, 256)
generator_model_name = "mod_base_4upscl"
generator_weights_path = r"training_data\dcgan\mod_base_4upscl__mod_ext_5layers__5pt\weights\410\generator_mod_base_4upscl.h5"
discriminator_model_name = "mod_ext_5layers"
discriminator_weights_path = r"training_data\dcgan\mod_base_4upscl__mod_ext_5layers__5pt\weights\410\discriminator_mod_ext_5layers.h5"
discriminator_realness_threshold = 0.95

num_of_images = 20
image_save_path = "generated_images"

lat_input = Input(shape=(latent_dim,))
preq_gen = getattr(generator_models_spreadsheet, generator_model_name)(lat_input, img_shape, img_shape[2])
gen_mod = Model(lat_input, preq_gen, name="generator")
disc_mod = None
if isinstance(discriminator_model_name, str):
	image_input = Input(shape=img_shape)
	preq_disc = getattr(discriminator_models_spreadsheet, discriminator_model_name)(image_input)
	preq_disc = Dense(1, activation="sigmoid")(preq_disc)
	disc_mod = Model(image_input, preq_disc, name="discriminator")

if generator_weights_path: gen_mod.load_weights(generator_weights_path)
if discriminator_weights_path and disc_mod: disc_mod.load_weights(discriminator_weights_path)

gen_images = []
def generate_images():
	global gen_images

	while len(gen_images) != num_of_images:
		missing_images = num_of_images - len(gen_images)
		noise = np.random.normal(np.random.normal(0.0, 1.0, size=(missing_images, latent_dim)))
		images = gen_mod.predict(noise)

		if disc_mod:
			predicted_realnesses = disc_mod.predict(images)[0]
			for image, realness in zip(images, predicted_realnesses):
				if realness >= discriminator_realness_threshold:
					gen_images.append(image)
		else:
			for image in images:
				gen_images.append(image)
				
generate_images()
gen_images = np.array(gen_images)
gen_images = (0.5 * gen_images + 0.5) * 255

if os.path.exists(image_save_path): shutil.rmtree(image_save_path, True)
os.makedirs(image_save_path)

for idx, image in enumerate(gen_images):
	image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
	if img_shape[0] != final_size[0] and img_shape[0] != final_size[1]:
		image = cv.resize(image, final_size, cv.INTER_CUBIC)
	cv.imwrite(f"{image_save_path}/img_{idx+1}.png", image)