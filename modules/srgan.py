import colorama
from colorama import Fore
import os
from typing import Union
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Optimizer, Adam
from keras.layers import Input, Dense
from keras.models import Model
from keras.engine.network import Network
from keras.applications.vgg19 import VGG19
from keras.initializers import RandomNormal
from keras.utils import plot_model
from statistics import mean
from PIL import Image
import numpy as np
import cv2 as cv
from collections import deque
from tqdm import tqdm
import json
import random
import time
import imagesize
from multiprocessing.pool import ThreadPool

from modules.models import upscaling_generator_models_spreadsheet, discriminator_models_spreadsheet
from modules.custom_tensorboard import TensorBoardCustom
from modules.batch_maker import BatchMaker
from modules.helpers import time_to_format

tf.get_logger().setLevel('ERROR')
colorama.init()

# Calculate start image size based on final image size and number of upscales
def count_upscaling_start_size(target_image_shape: tuple, num_of_upscales: int):
	upsc = (target_image_shape[0] // (2 ** num_of_upscales), target_image_shape[1] // (2 ** num_of_upscales), target_image_shape[2])
	if upsc[0] < 1 or upsc[1] < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
	return upsc

class VGG_LOSS(object):
	def __init__(self, image_shape):
		self.image_shape = image_shape

	# computes VGG loss or content loss
	def vgg_loss(self, y_true, y_pred):
		vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
		vgg19.trainable = False
		for l in vgg19.layers:
			l.trainable = False
		model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
		model.trainable = False

		return K.mean(K.square(model(y_true) - model(y_pred)))

class SRGAN:
	AGREGATE_STAT_INTERVAL = 1  # Interval of saving data
	RESET_SEEDS_INTERVAL = 10  # Interval of checking norm gradient value of combined model
	CHECKPOINT_SAVE_INTERVAL = 1  # Interval of saving checkpoint

	def __init__(self, dataset_path:str, num_of_upscales:int,
	             gen_mod_name: str, disc_mod_name: str,
	             training_progress_save_path:str=None,
	             generator_optimizer: Optimizer = Adam(0.0002, 0.5), discriminator_optimizer: Optimizer = Adam(0.0002, 0.5),
	             discriminator_label_noise: float = None, discriminator_label_noise_decay: float = None, discriminator_label_noise_min: float = 0.001,
	             batch_size: int = 32, buffered_batches:int=20,
	             generator_weights: Union[str, None, int] = None, discriminator_weights: Union[str, None, int] = None,
	             start_episode: int = 0, load_from_checkpoint: bool = False):

		self.disc_mod_name = disc_mod_name
		self.gen_mod_name = gen_mod_name
		self.num_of_upscales = num_of_upscales

		self.discriminator_label_noise = discriminator_label_noise
		self.discriminator_label_noise_decay = discriminator_label_noise_decay
		self.discriminator_label_noise_min = discriminator_label_noise_min

		self.batch_size = batch_size

		if start_episode < 0: start_episode = 0
		self.epoch_counter = start_episode

		# Create array of input image paths
		self.train_data = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
		self.data_length = len(self.train_data)

		# Load one image to get shape of it
		self.target_image_shape = cv.imread(self.train_data[0]).shape

		# Check image size validity
		if self.target_image_shape[0] < 4 or self.target_image_shape[1] < 4: raise Exception("Images too small, min size (4, 4)")

		# Starting image size calculate
		self.start_image_shape = count_upscaling_start_size(self.target_image_shape, self.num_of_upscales)

		# Check validity of whole datasets
		self.validate_dataset()

		# Initialize training data folder and logging
		self.training_progress_save_path = training_progress_save_path
		self.tensorboard = None
		if self.training_progress_save_path:
			self.training_progress_save_path = os.path.join(self.training_progress_save_path, f"{self.gen_mod_name}__{self.disc_mod_name}__{self.start_image_shape}_to_{self.target_image_shape}")
			self.tensorboard = TensorBoardCustom(log_dir=os.path.join(self.training_progress_save_path, "logs"))

		# Create array of input image paths
		self.train_data = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
		self.data_length = len(self.train_data)

		# Define static vars
		self.kernel_initializer = RandomNormal(stddev=0.02)
		self.progress_test_image_path = random.choice(self.train_data)

		# Create batchmaker and start it
		self.batch_maker = BatchMaker(self.train_data, self.data_length, self.batch_size, buffered_batches=buffered_batches)
		self.batch_maker.start()

		# Loss object
		self.loss_object = VGG_LOSS(self.target_image_shape)

		#################################
		###   Create discriminator    ###
		#################################
		self.discriminator = self.build_discriminator(disc_mod_name)
		self.discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer)
		print("\nDiscriminator Sumary:")
		self.discriminator.summary()

		#################################
		###     Create generator      ###
		#################################
		self.generator = self.build_generator(gen_mod_name)
		if self.generator.output_shape[1:] != self.target_image_shape: raise Exception("Invalid image input size for this generator model")
		print("\nGenerator Sumary:")
		self.generator.summary()

		#################################
		### Create combined generator ###
		#################################
		small_image_input = Input(shape=self.start_image_shape, name="small_image_input")
		gen_images = self.generator(small_image_input)

		# Create frozen version of discriminator
		frozen_discriminator = Network(self.discriminator.inputs, self.discriminator.outputs, name="frozen_discriminator")
		frozen_discriminator.trainable = False

		# Discriminator takes images and determinates validity
		valid = frozen_discriminator(gen_images)

		# Combine models
		# Train generator to fool discriminator
		self.combined_generator_model = Model(small_image_input, outputs=[gen_images, valid], name="srgan_model")
		self.combined_generator_model.compile(loss=[self.loss_object.vgg_loss, "binary_crossentropy"],
		                                      loss_weights=[1., 1e-3],
		                                      optimizer=generator_optimizer)

		# Load checkpoint
		if load_from_checkpoint: self.load_checkpoint()

		# Load weights from param and override checkpoint weights
		if generator_weights: self.generator.load_weights(generator_weights)
		if discriminator_weights: self.discriminator.load_weights(discriminator_weights)

		# Create some proprietary objects
		self.gen_labels = np.ones(shape=(self.batch_size, 1))

	# Check if datasets have consistent shapes
	def validate_dataset(self):
		def check_image(image_path):
			im_shape = imagesize.get(image_path)
			if im_shape[0] != self.target_image_shape[0] or im_shape[1] != self.target_image_shape[1]:
				return False
			return True

		print("Checking dataset validity")
		with ThreadPool(processes=8) as p:
			res = p.map(check_image, self.train_data)
			if not all(res): raise Exception("Inconsistent dataset")

		print("Dataset valid")

	# Create generator based on template selected by name
	def build_generator(self, model_name:str):
		small_image_input = Input(shape=self.start_image_shape)

		try:
			m = getattr(upscaling_generator_models_spreadsheet, model_name)(small_image_input, self.start_image_shape, self.num_of_upscales, self.kernel_initializer)
		except Exception as e:
			raise Exception(f"Generator model not found!\n{e}")

		return Model(small_image_input, m, name="generator_model")

	# Create discriminator based on teplate selected by name
	def build_discriminator(self, model_name:str, classification:bool=True):
		img = Input(shape=self.target_image_shape)

		try:
			m = getattr(discriminator_models_spreadsheet, model_name)(img, self.kernel_initializer)
		except Exception as e:
			raise Exception(f"Discriminator model not found!\n{e}")

		if classification:
			m = Dense(1, activation="sigmoid")(m)

		return Model(img, m, name="discriminator_model")

	def train(self, epochs: int,
	          progress_images_save_interval: int = None, weights_save_interval: int = None,
	          discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False):

		# Check arguments and input data
		if self.training_progress_save_path is not None and progress_images_save_interval is not None and progress_images_save_interval <= epochs and epochs % progress_images_save_interval != 0: raise Exception("Invalid progress save interval")
		if weights_save_interval is not None and weights_save_interval <= epochs and epochs % weights_save_interval != 0: raise Exception("Invalid weights save interval")
		if self.train_data is None: raise Exception("No datasets loaded")

		if self.training_progress_save_path:
			if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)

		# Training variables
		num_of_batches = self.data_length // self.batch_size
		end_epoch = self.epoch_counter + epochs

		epochs_time_history = deque(maxlen=5)

		print(Fore.GREEN + f"Starting training on epoch {self.epoch_counter}" + Fore.RESET)
		for _ in range(epochs):
			ep_start = time.time()
			for _ in tqdm(range(num_of_batches), unit="batches", smoothing=0.5, leave=False):
				large_images, small_images = self.batch_maker.get_batch_resized_and_original(self.start_image_shape)

				gen_imgs = self.generator.predict(small_images)

				# Train discriminator (real as ones and fake as zeros)
				if discriminator_smooth_real_labels:
					disc_real_labels = np.random.uniform(0.7, 1.2, size=(self.batch_size, 1))
				else:
					disc_real_labels = np.ones(shape=(self.batch_size, 1))

				if discriminator_smooth_fake_labels:
					disc_fake_labels = np.random.uniform(0, 0.3, size=(self.batch_size, 1))
				else:
					disc_fake_labels = np.zeros(shape=(self.batch_size, 1))

				self.discriminator.train_on_batch(large_images, disc_real_labels)
				self.discriminator.train_on_batch(gen_imgs, disc_fake_labels)

				### Train Generator ###
				# Train generator (wants discriminator to recognize fake images as valid)
				large_images, small_images = self.batch_maker.get_batch_resized_and_original(self.start_image_shape)
				self.combined_generator_model.train_on_batch(small_images, [large_images, self.gen_labels])

			time.sleep(0.5)
			self.epoch_counter += 1
			epochs_time_history.append(time.time() - ep_start)
			if self.tensorboard:
				self.tensorboard.step = self.epoch_counter

			# Decay label noise
			if self.discriminator_label_noise_decay:
				self.discriminator_label_noise = max([self.discriminator_label_noise_min, (self.discriminator_label_noise * self.discriminator_label_noise_decay)])

				if (self.discriminator_label_noise_min == 0) and (self.discriminator_label_noise != 0) and (self.discriminator_label_noise < 0.005):
					self.discriminator_label_noise = 0

			# Seve stats and print them to console
			if self.epoch_counter % self.AGREGATE_STAT_INTERVAL == 0:
				large_images, small_images = self.batch_maker.get_batch_resized_and_original(self.start_image_shape)
				gen_imgs = self.generator.predict(small_images)

				# Evaluate models state
				disc_real_loss = self.discriminator.test_on_batch(large_images, np.ones(shape=(large_images.shape[0], 1)))
				disc_fake_loss = self.discriminator.test_on_batch(gen_imgs, np.zeros(shape=(gen_imgs.shape[0], 1)))
				vgg_loss, gen_loss = self.combined_generator_model.test_on_batch(small_images, [large_images, np.ones(shape=(large_images.shape[0], 1))])

				if self.tensorboard:
					self.tensorboard.log_kernels_and_biases(self.generator)
					self.tensorboard.update_stats(self.epoch_counter, disc_real_loss=disc_real_loss, disc_fake_loss=disc_fake_loss, gen_loss=gen_loss, vgg_loss=vgg_loss, disc_label_noise=self.discriminator_label_noise if self.discriminator_label_noise else 0)

				print(Fore.GREEN + f"{self.epoch_counter}/{end_epoch}, Remaining: {time_to_format(mean(epochs_time_history) * (end_epoch - self.epoch_counter))} - [D-R loss: {round(float(disc_real_loss), 5)}, D-F loss: {round(float(disc_fake_loss), 5)}] [G loss: {round(float(gen_loss), 5)}, G VGG loss: {round(float(vgg_loss), 5)}] - Epsilon: {round(self.discriminator_label_noise, 4)}" + Fore.RESET)

			# Save progress
			if self.training_progress_save_path is not None and progress_images_save_interval is not None and self.epoch_counter % progress_images_save_interval == 0:
				self.__save_img()

			# Save weights of models
			if weights_save_interval is not None and self.epoch_counter % weights_save_interval == 0:
				self.save_weights()

			# Save checkpoint
			if self.epoch_counter % self.CHECKPOINT_SAVE_INTERVAL == 0:
				self.save_checkpoint()
				print(Fore.BLUE + "Checkpoint created" + Fore.RESET)

			# Reset seeds
			if self.epoch_counter % self.RESET_SEEDS_INTERVAL == 0:
				# Change seed for keeping as low number of constants as possible
				np.random.seed(None)
				random.seed()

		# Shutdown helper threads
		print(Fore.GREEN + "Training Complete - Waiting for other threads to finish" + Fore.RESET)
		self.batch_maker.terminate = True
		self.save_checkpoint()
		self.save_weights()
		self.batch_maker.join()
		print(Fore.GREEN + "All threads finished" + Fore.RESET)

	def __save_img(self):
		if not os.path.exists(self.training_progress_save_path + "/progress_images"): os.makedirs(self.training_progress_save_path + "/progress_images")
		gen_img = self.generator.predict(cv.cvtColor(cv.imread(self.progress_test_image_path), cv.COLOR_BGR2RGB) / 127.5 - 1.0)

		# Rescale images 0 to 255
		gen_img = (0.5 * gen_img + 0.5) * 255

		gen_img = cv.cvtColor(gen_img, cv.COLOR_BGR2RGB)
		cv.imwrite(f"{self.training_progress_save_path}/progress_images/{self.epoch_counter}.png", gen_img)

	def save_weights(self):
		save_dir = self.training_progress_save_path + "/weights/" + str(self.epoch_counter)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")
		self.discriminator.save_weights(f"{save_dir}/discriminator_{self.disc_mod_name}.h5")

	def save_models_structure_images(self):
		save_path = self.training_progress_save_path + "/model_structures"
		if not os.path.exists(save_path): os.makedirs(save_path)
		plot_model(self.combined_generator_model, os.path.join(save_path, "combined.png"), expand_nested=True, show_shapes=True)
		plot_model(self.generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)
		plot_model(self.discriminator, os.path.join(save_path, "discriminator.png"), expand_nested=True, show_shapes=True)

	def load_checkpoint(self):
		checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
		if not os.path.exists(os.path.join(checkpoint_base_path, "checkpoint_data.json")): return

		with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "rb") as f:
			data = json.load(f)

			if data:
				self.epoch_counter = int(data["episode"])
				self.generator.load_weights(data["gen_path"])
				self.discriminator.load_weights(data["disc_path"])
				if data["disc_label_noise"]:
					self.discriminator_label_noise = float(data["disc_label_noise"])
				self.progress_test_image_path = data["test_image"]

	def save_checkpoint(self):
		checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
		if not os.path.exists(checkpoint_base_path): os.makedirs(checkpoint_base_path)

		self.generator.save_weights(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5")
		self.discriminator.save_weights(f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5")

		data = {
			"episode": self.epoch_counter,
			"gen_path": f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5",
			"disc_path": f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5",
			"disc_label_noise": self.discriminator_label_noise,
			"test_image": self.progress_test_image_path
		}

		with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "w", encoding='utf-8') as f:
			json.dump(data, f)

	def make_progress_gif(self, frame_duration:int=16):
		if not os.path.exists(self.training_progress_save_path + "/progress_images"): return
		if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)

		frames = []
		img_file_names = os.listdir(self.training_progress_save_path + "/progress_images")

		for im_file in img_file_names:
			if os.path.isfile(self.training_progress_save_path + "/progress_images/" + im_file):
				frames.append(Image.open(self.training_progress_save_path + "/progress_images/" + im_file))

		if len(frames) > 2:
			frames[0].save(f"{self.training_progress_save_path}/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, optimize=False, duration=frame_duration, loop=0)