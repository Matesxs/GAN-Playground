import os
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import Adam, Optimizer
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Reshape, Dense, Flatten, BatchNormalization, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.utils import plot_model
import tensorflow as tf
from PIL import Image
from collections import deque
import math
import cv2 as cv
import time
from tqdm import tqdm
from typing import Union

from src.batch_maker import BatchMaker

tf.get_logger().setLevel('ERROR')

class DCGAN:
	def __init__(self, train_images:Union[np.ndarray, list, None, str], optimizer:Optimizer=Adam(0.0002, 0.5), latent_dim:int=100, progress_image_path:str=None, ex:int=5, gen_v:int=1, disc_v:int=1, disc_weights:str=None, gen_weights:str=None):
		self.optimizer = optimizer
		self.latent_dim = latent_dim
		self.ex = ex
		self.epoch_counter = 0
		self.progress_image_path = progress_image_path

		if type(train_images) == list:
			self.train_data = np.array(train_images)
			self.data_length = self.train_data.shape[0]
		elif train_images is None:
			(x_train, y_train), (x_test, y_test) = cifar10.load_data()
			# Selecting cats :)
			x_train = x_train[np.where(y_train == 3)[0]]
			x_test = x_test[np.where(y_test == 3)[0]]
			x_train = np.concatenate((x_train, x_test))
			self.train_data = x_train
			self.data_length = self.train_data.shape[0]
		elif type(train_images) == str:
			self.train_data = [os.path.join(train_images, file) for file in os.listdir(train_images)]
			self.data_length = len(self.train_data)
		else:
			self.train_data = train_images
			self.data_length = self.train_data.shape[0]

		if type(train_images) != str:
			# Scale -1 to 1
			self.train_data = self.train_data / 127.5 - 1.0
			self.image_shape = self.train_data[0].shape
		else:
			tmp_image = cv.imread(self.train_data[0])
			self.image_shape = tmp_image.shape
		self.image_channels = self.image_shape[2]

		# Check image size validity
		if self.image_shape[0] != self.image_shape[1]: raise Exception("Images must be squared")
		if self.image_shape[0] < 4: raise Exception("Images too small")
		if not math.log2(self.image_shape[0]).is_integer(): raise Exception("Invalid size, size have to be power of 2")

		# Check validity of dataset
		self.validate_dataset()

		# Define static vars
		self.static_noise = np.random.normal(size=(self.ex*self.ex, self.latent_dim))
		self.conv_kerner_initializer = RandomNormal(stddev=0.02)

		# Build discriminator
		self.discriminator = self.build_discriminator(disc_v)
		self.discriminator.compile(loss="binary_crossentropy", optimizer=self.optimizer,  metrics=['accuracy'])
		if disc_weights: self.discriminator.load_weights(disc_weights)

		# Build generator
		self.generator = self.build_generator(gen_v)
		if gen_weights: self.generator.load_weights(gen_weights)

		# Generator takes noise and generates images
		noise_input = Input(shape=(self.latent_dim,), name="noise_input")
		gen_images = self.generator(noise_input)

		# For combined model we will only train generator
		self.discriminator.trainable = False

		# Discriminator takes images and determinates validity
		valid = self.discriminator(gen_images)

		# Combine models
		# Train generator to fool discriminator
		self.combined_model = Model(noise_input, valid)
		self.combined_model.compile(loss="binary_crossentropy", optimizer=self.optimizer)

		# Statistics
		self.gen_losses = deque()
		self.disc_losses = deque()

	def validate_dataset(self):
		if type(self.train_data) == list:
			for im_path in self.train_data:
				im_shape = cv.imread(im_path).shape
				if im_shape != self.image_shape:
					raise Exception("Inconsistent dataset")
		else:
			for image in self.train_data:
				if image.shape != self.image_shape:
					raise Exception("Inconsistent dataset")
		print("Dataset valid")

	def count_upscaling_start_size(self, num_of_upscales:int):
		upsc = self.image_shape[0] / (2**num_of_upscales)
		if upsc < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
		return int(upsc)

	def build_generator(self, version:int=1):
		model = Sequential(name="generator_submodel")

		if version == 1:
			st_s = self.count_upscaling_start_size(2)

			# (256 * st_s^2,) -> (st_s, st_s, 256)
			model.add(Dense(256 * st_s * st_s, input_shape=(self.latent_dim,), kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU())
			model.add(Reshape((st_s, st_s, 256)))

			# (st_s, st_s, 256) -> (st_s, st_s, 256)
			model.add(Conv2DTranspose(256, (5, 5), strides=(1, 1), padding="same", kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU())

			# (st_s, st_s, 256) -> (2*st_s, 2*st_s, 128)
			model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU())

			# (2*st_s, 2*st_s, 128) -> (4*st_s, 4*st_s, num_ch)
			model.add(Conv2DTranspose(self.image_channels, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.conv_kerner_initializer, activation="tanh"))
		elif version == 2:
			st_s = self.count_upscaling_start_size(4)

			# (256*st_s^2,) -> (st_s, st_s, 256)
			model.add(Dense(st_s * st_s * 256, input_shape=(self.latent_dim,), kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))
			model.add(Reshape((st_s, st_s, 256)))

			# (st_s, st_s, 256) -> (2*st_s, 2*st_s, 128)
			model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (2*st_s, 2*st_s, 128) -> (4*st_s, 4*st_s, 64)
			model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (4*st_s, 4*st_s, 64) -> (8*st_s, 8*st_s, 32)
			model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (8*st_s, 8*st_s, 32) -> (16*st_s, 16*st_s, num_ch)
			model.add(Conv2DTranspose(self.image_channels, (5, 5), strides=(2, 2), padding="same", activation="tanh"))
		elif version == 3:
			st_s = self.count_upscaling_start_size(3)

			# (512 * st_s^2,) -> (st_s, st_s, 512)
			model.add(Dense(512 * st_s * st_s, input_shape=(self.latent_dim,), kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))
			model.add(Reshape((st_s, st_s, 512)))

			# (st_s, st_s, 512) -> (2*st_s, 2*st_s, 256)
			model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (2*st_s, 2*st_s, 256) -> (4*st_s, 4*st_s, 128)
			model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (4*st_s, 4*st_s, 128) -> (8*st_s, 8*st_s, num_ch)
			model.add(Conv2DTranspose(self.image_channels, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.conv_kerner_initializer, activation="tanh"))
		else:
			raise Exception("Generator invalid version")

		print("\nGenerator Sumary:")
		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self, version:int=1):
		model = Sequential(name="discriminator_submodel")

		if version == 1:
			model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=self.image_shape, kernel_initializer=self.conv_kerner_initializer))
			model.add(LeakyReLU(0.2))

			model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			model.add(Conv2D(256, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			model.add(Conv2D(512, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			model.add(Flatten())
		elif version == 2:
			model.add(Conv2D(32, (5, 5), padding='same', strides=(2, 2), input_shape=self.image_shape, kernel_initializer=self.conv_kerner_initializer))
			model.add(LeakyReLU(0.2))

			model.add(Conv2D(64, (5, 5), padding='same', strides=(2, 2)))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			model.add(Conv2D(128, (5, 5), padding='same', strides=(2, 2)))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			model.add(Conv2D(256, (5, 5), padding='same', strides=(2, 2)))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			model.add(Flatten())
		else:
			raise Exception("Discriminator invalid version")

		model.add(Dense(1, activation="sigmoid"))

		print("\nDiscriminator Sumary:")
		model.summary()

		img = Input(shape=self.image_shape)
		validity = model(img)

		return Model(img, validity, name="discriminator_model")

	def train(self, epochs:int=200, batch_size:int=64, save_interval:int=None, smooth:float=0.1, trick_fake:bool=False):
		if self.progress_image_path is not None and save_interval is not None:
			if epochs%save_interval != 0: raise Exception("Invalid save interval")

		# Create batchmaker and start it
		batch_maker = BatchMaker(self.train_data, self.data_length, batch_size)
		batch_maker.start()

		# Validity arrays
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		g_loss, d_loss = None, None

		s_time = time.time()
		for _ in tqdm(range(epochs), unit="ep"):
			for batch in range(self.data_length // batch_size):
				### Train Discriminator ###
				# Select batch of valid images
				imgs = batch_maker.get_batch()

				# Sample noise and generate new images
				noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
				gen_imgs = self.generator.predict(noise)

				# Train discriminator (real as ones and fake as zeros)
				self.discriminator.trainable = True
				d_loss_real = self.discriminator.train_on_batch(imgs, valid * (1.0 - smooth))
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
				self.discriminator.trainable = False
				d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
				if d_loss < 0: d_loss = 0.0

				# Calling destructor of loaded images
				del imgs
				del gen_imgs

				### Train Generator ###
				# Train generator (wants discriminator to recognize fake images as valid)
				if not trick_fake:
					g_loss = self.combined_model.train_on_batch(noise, valid)
				else:
					trick = np.random.uniform(0.7, 1.4, size=(batch_size, 1))
					g_loss = self.combined_model.train_on_batch(noise, trick)
				if g_loss < 0: g_loss = 0.0

			# Save statistics
			self.gen_losses.append(g_loss)
			self.disc_losses.append(d_loss)

			# Save progress
			if self.progress_image_path is not None and save_interval is not None and (self.epoch_counter + 1) % save_interval == 0:
				print(f"[D loss: {d_loss}] [G loss: {g_loss}] - Elapsed: {round((time.time() - s_time) / 60, 1)}min")
				self.__save_imgs(self.epoch_counter)
			self.epoch_counter += 1

		# Shutdown batchmaker and wait for its exit
		batch_maker.terminate = True
		batch_maker.join()

	def __save_imgs(self, epoch):
		if not os.path.isdir(self.progress_image_path): os.mkdir(self.progress_image_path)
		gen_imgs = self.generator.predict(self.static_noise)

		# Rescale images 0 to 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(self.ex, self.ex)
		cnt = 0
		for i in range(self.ex):
			for j in range(self.ex):
				if self.image_channels == 3:
					axs[i, j].imshow(gen_imgs[cnt])
				else:
					axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
				axs[i, j].axis('off')
				cnt += 1

		fig.savefig(f"{self.progress_image_path}/{epoch + 1}.png")
		plt.close()

	def show_current_state(self, num_of_states:int=1, ex:int=3):
		for _ in range(num_of_states):
			gen_imgs = self.generator.predict(np.random.normal(size=(ex * ex, self.latent_dim)))

			# Rescale images 0 to 1
			gen_imgs = 0.5 * gen_imgs + 0.5

			fig, axs = plt.subplots(ex, ex)
			cnt = 0
			for i in range(ex):
				for j in range(ex):
					if self.image_channels == 3:
						axs[i, j].imshow(gen_imgs[cnt])
					else:
						axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
					axs[i, j].axis('off')
					cnt += 1
			plt.show()
			plt.close()

	def show_sample_of_dataset(self):
		fig, axs = plt.subplots(self.ex, self.ex)
		cnt = 0
		for i in range(self.ex):
			for j in range(self.ex):
				if type(self.train_data) != list:
					if self.image_channels == 3:
						axs[i, j].imshow(self.train_data[np.random.randint(0, self.data_length, size=1)][0])
					else:
						axs[i, j].imshow(self.train_data[np.random.randint(0, self.data_length, size=1), :, :, 0][0], cmap="gray")
				else:
					if self.image_channels == 3:
						axs[i, j].imshow(cv.imread(self.train_data[np.random.randint(0, self.data_length, size=1)[0]]))
					else:
						axs[i, j].imshow(cv.imread(self.train_data[np.random.randint(0, self.data_length, size=1)[0]])[:, :, 0], cmap="gray")
				axs[i, j].axis('off')
				cnt += 1
		plt.show()
		plt.close()

	def generate_random_images(self, number_of_images:int=5, save_path:str="."):
		if not os.path.isdir(save_path): os.mkdir(save_path)
		gen_imgs = self.generator.predict(np.random.normal(size=(number_of_images, self.latent_dim)))

		# Rescale images 0 to 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		for idx, image in enumerate(gen_imgs):
			cv.imwrite(f"{save_path}/gen_im_{idx}.png", image)

	def show_training_stats(self):
		plt.plot(self.disc_losses)
		plt.plot(self.gen_losses)
		plt.legend(["Disc Loss", "Gen Loss"])
		plt.show()
		plt.close()

	def plot_models(self, save_path:str="."):
		if not os.path.isdir(save_path): os.mkdir(save_path)
		plot_model(self.combined_model, "combined.png", expand_nested=True)
		plot_model(self.generator, "generator.png", expand_nested=True)
		plot_model(self.discriminator, "discriminator.png", expand_nested=True)

	def clear_output_folder(self):
		if not os.path.isdir(self.progress_image_path): return

		img_file_names = os.listdir(self.progress_image_path)
		for im_file in img_file_names:
			if os.path.isfile(self.progress_image_path + "/" + im_file):
				os.remove(self.progress_image_path + "/" + im_file)

	def save_weights(self, save_directory:str= "."):
		if not os.path.isdir(save_directory): os.mkdir(save_directory)
		save_dir = os.path.join(save_directory, str(self.epoch_counter))
		if not os.path.isdir(save_dir): os.mkdir(save_dir)
		self.generator.save_weights(f"{save_dir}/generator.h5")
		self.discriminator.save_weights(f"{save_dir}/discriminator.h5")

	def save_weights_prompt(self, save_directory:str= "."):
		while True:
			ans = input("Do you want to save models weights? (y/n)\n")
			if ans == "y":
				self.save_weights(save_directory)
				break
			elif ans == "n":
				break

	def make_gif(self, path:str=None, duration:int=120):
		if not os.path.isdir(self.progress_image_path): return
		if not path: path = f"{self.progress_image_path}"
		if not os.path.isdir(path): os.mkdir(path)

		frames = []
		img_file_names = os.listdir(self.progress_image_path)

		for im_file in img_file_names:
			if os.path.isfile(self.progress_image_path + "/" + im_file):
				frames.append(Image.open(self.progress_image_path + "/" + im_file))

		if len(frames) > 2:
			frames[0].save(f"{path}/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=0)