import os
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import Adam, Optimizer
from keras.models import Model
from keras.layers import Input, Dense
from keras.initializers import RandomNormal
from keras.utils import plot_model
import tensorflow as tf
from PIL import Image
from collections import deque
import math
import cv2 as cv
from tqdm import tqdm
from typing import Union

from src.batch_maker import BatchMaker
from src import generator_models_spreadsheet
from src import discriminator_models_spreadsheet

tf.get_logger().setLevel('ERROR')

class DCGAN:
	def __init__(self, train_images:Union[np.ndarray, list, None, str], generator_optimizer:Optimizer=Adam(0.0002, 0.5), discriminator_optimizer:Optimizer=Adam(0.0002, 0.5), latent_dim:int=100, progress_image_path:str=None, progress_image_num:int=5, gen_mod_name:str= "mod_base_2upscl", disc_mod_name:str= "mod_base_4layers", discriminator_weights:str=None, generator_weights:str=None):
		self.generator_optimizer = generator_optimizer
		self.discriminator_optimizer = discriminator_optimizer

		self.disc_mod_name = disc_mod_name
		self.gen_mod_name = gen_mod_name

		self.latent_dim = latent_dim
		self.progress_image_num = progress_image_num
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
		self.static_noise = np.random.normal(0.0, 1.0, size=(self.progress_image_num * self.progress_image_num, self.latent_dim))
		self.kernel_initializer = RandomNormal(stddev=0.02)

		# Build discriminator
		self.discriminator = self.build_discriminator(disc_mod_name)
		self.discriminator.compile(loss="binary_crossentropy", optimizer=self.discriminator_optimizer, metrics=['accuracy'])
		if discriminator_weights: self.discriminator.load_weights(discriminator_weights)

		# Build generator
		self.generator = self.build_generator(gen_mod_name)
		if generator_weights: self.generator.load_weights(generator_weights)

		# Generator takes noise and generates images
		noise_input = Input(shape=(self.latent_dim,), name="noise_input")
		gen_images = self.generator(noise_input)

		# For combined model we will only train generator
		self.discriminator.trainable = False

		# Discriminator takes images and determinates validity
		valid = self.discriminator(gen_images)

		# Combine models
		# Train generator to fool discriminator
		self.combined_model = Model(noise_input, valid, name="dcgan_model")
		self.combined_model.compile(loss="binary_crossentropy", optimizer=self.generator_optimizer)

		# Statistics
		self.gen_losses = deque()
		self.gen_mean_losses = deque()
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

	def build_generator(self, model_name:str="mod_base_2upscl"):
		noise = Input(shape=(self.latent_dim,))

		try:
			m = getattr(generator_models_spreadsheet, model_name)(noise, self.image_shape, self.image_channels, self.kernel_initializer)
		except Exception as e:
			raise Exception(f"Generator model not found!\n{e.__traceback__}")

		model = Model(noise, m, name="generator_model")

		print("\nGenerator Sumary:")
		model.summary()

		return model

	def build_discriminator(self, model_name:str="mod_base_4layers"):
		img = Input(shape=self.image_shape)

		try:
			m = getattr(discriminator_models_spreadsheet, model_name)(img, self.kernel_initializer)
		except Exception as e:
			raise Exception(f"Discriminator model not found!\n{e.__traceback__}")

		m = Dense(1, activation="sigmoid")(m)

		model = Model(img, m, name="discriminator_model")

		print("\nDiscriminator Sumary:")
		model.summary()

		return model

	def train(self, epochs:int=200, batch_size:int=64, progress_save_interval:int=None, weights_save_interval:int=None, weights_save_path:str=None, disc_train_multip:int=1, generator_smooth_labels:bool=False, discriminator_smooth_labels:bool=False):
		if self.progress_image_path is not None and progress_save_interval is not None and progress_save_interval <= epochs:
			if epochs%progress_save_interval != 0: raise Exception("Invalid progress save interval")
		if weights_save_path is not None and weights_save_interval is not None and weights_save_interval <= epochs:
			if epochs%weights_save_interval != 0: raise Exception("Invalid weights save interval")
		if disc_train_multip < 1: raise Exception("Invalid discriminator training multiplier")
		if self.data_length < batch_size: raise Exception("Invalid batch size")

		# Create batchmaker and start it
		batch_maker = BatchMaker(self.train_data, self.data_length, batch_size)
		batch_maker.start()

		num_of_batches = self.data_length // batch_size
		g_loss, d_loss = None, None
		disc_batch_losses = deque(maxlen=num_of_batches)
		gen_batch_losses = deque(maxlen=num_of_batches)

		for _ in tqdm(range(epochs), unit="ep"):
			for batch in range(num_of_batches):
				for _ in range(disc_train_multip):
					### Train Discriminator ###
					# Select batch of valid images
					imgs = batch_maker.get_batch()

					# Sample noise and generate new images
					gen_imgs = self.generator.predict(np.random.normal(0.0, 1.0, (batch_size, self.latent_dim)))

					# Train discriminator (real as ones and fake as zeros)
					if discriminator_smooth_labels:
						disc_real_labels = np.random.uniform(0.7, 1.2, size=(batch_size, 1))
						disc_fake_labels = np.random.uniform(0.0, 0.2, size=(batch_size, 1))
					else:
						disc_real_labels = np.ones(shape=(batch_size, 1))
						disc_fake_labels = np.zeros(shape=(batch_size, 1))

					self.discriminator.trainable = True
					d_stat_real = self.discriminator.train_on_batch(imgs, disc_real_labels)
					d_stat_fake = self.discriminator.train_on_batch(gen_imgs, disc_fake_labels)
					self.discriminator.trainable = False

					d_loss = 0.5 * (d_stat_real[0] + d_stat_fake[0])
					if d_loss < 0: d_loss = 0.0
					disc_batch_losses.append(d_loss)

				### Train Generator ###
				# Train generator (wants discriminator to recognize fake images as valid)
				if generator_smooth_labels:
					gen_labels = np.random.uniform(0.7, 1.2, size=(batch_size, 1))
				else:
					gen_labels = np.ones(shape=(batch_size, 1))

				g_loss = self.combined_model.train_on_batch(np.random.normal(0.0, 1.0, (batch_size, self.latent_dim)), gen_labels)

				if g_loss < 0: g_loss = 0.0
				gen_batch_losses.append(g_loss)

			# Save statistics
			gen_loss = np.mean(np.array(gen_batch_losses))
			self.gen_losses.append(gen_loss)
			mean_gen_loss = np.mean(np.array(self.gen_losses)[-100:])
			self.gen_mean_losses.append(mean_gen_loss)
			disc_loss = np.mean(np.array(disc_batch_losses))
			self.disc_losses.append(disc_loss)

			# Save progress
			if self.progress_image_path is not None and progress_save_interval is not None and (self.epoch_counter + 1) % progress_save_interval == 0:
				print(f"[D loss: {disc_loss}] [G loss: {gen_loss}, Mean G Loss: {mean_gen_loss}]")
				self.__save_imgs(self.epoch_counter)

			if weights_save_path is not None and weights_save_path is not None and (self.epoch_counter + 1) % weights_save_interval == 0:
				self.save_weights(weights_save_path)

			self.epoch_counter += 1

		# Shutdown batchmaker and wait for its exit
		batch_maker.terminate = True
		batch_maker.join()

	def __save_imgs(self, epoch):
		if not os.path.isdir(self.progress_image_path): os.mkdir(self.progress_image_path)
		gen_imgs = self.generator.predict(self.static_noise)

		# Rescale images 0 to 1
		gen_imgs = (0.5 * gen_imgs + 0.5) * 255

		final_image = np.zeros(shape=(self.image_shape[0] * self.progress_image_num, self.image_shape[0] * self.progress_image_num, self.image_channels)).astype(np.float32)

		cnt = 0
		for i in range(self.progress_image_num):
			for j in range(self.progress_image_num):
				cursor = (i * self.image_shape[0], j * self.image_shape[0])
				if self.image_channels == 3:
					final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[0] * j:self.image_shape[0] * (j + 1), :] = gen_imgs[cnt]
				else:
					final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[0] * j:self.image_shape[0] * (j + 1), 0] = gen_imgs[cnt, :, :, 0]
				cnt += 1
		final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
		cv.imwrite(f"{self.progress_image_path}/{self.epoch_counter + 1}.png", final_image)

	def show_current_state(self, num_of_states:int=1, progress_image_num:int=3):
		for _ in range(num_of_states):
			gen_imgs = self.generator.predict(np.random.normal(0.0, 1.0, size=(progress_image_num * progress_image_num, self.latent_dim)))

			# Rescale images 0 to 1
			gen_imgs = 0.5 * gen_imgs + 0.5

			fig, axs = plt.subplots(progress_image_num, progress_image_num)

			cnt = 0
			for i in range(progress_image_num):
				for j in range(progress_image_num):
					if self.image_channels == 3:
						axs[i, j].imshow(gen_imgs[cnt])
					else:
						axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
					axs[i, j].axis('off')
					cnt += 1
			plt.show()
			plt.close()

	def show_sample_of_dataset(self, progress_image_num:int=5):
		fig, axs = plt.subplots(progress_image_num, progress_image_num)

		cnt = 0
		for i in range(progress_image_num):
			for j in range(progress_image_num):
				if type(self.train_data) != list:
					if self.image_channels == 3:
						axs[i, j].imshow(self.train_data[np.random.randint(0, self.data_length, size=1)][0])
					else:
						axs[i, j].imshow(self.train_data[np.random.randint(0, self.data_length, size=1), :, :, 0][0], cmap="gray")
				else:
					if self.image_channels == 3:
						axs[i, j].imshow(cv.cvtColor(cv.imread(self.train_data[np.random.randint(0, self.data_length, size=1)[0]]), cv.COLOR_BGR2RGB))
					else:
						axs[i, j].imshow(cv.cvtColor(cv.imread(self.train_data[np.random.randint(0, self.data_length, size=1)[0]]), cv.COLOR_BGR2RGB)[:, :, 0], cmap="gray")
				axs[i, j].axis('off')
				cnt += 1
		plt.show()
		plt.close()

	def generate_random_images(self, number_of_images:int=5, save_path:str=".", blur:bool=False):
		if not os.path.isdir(save_path): os.mkdir(save_path)
		gen_imgs = self.generator.predict(np.random.normal(0.0, 1.0, size=(number_of_images, self.latent_dim)))

		# Rescale images 0 to 255
		gen_imgs = (0.5 * gen_imgs + 0.5) * 255

		for idx, image in enumerate(gen_imgs):
			if blur: image = cv.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
			image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
			cv.imwrite(f"{save_path}/gen_im_{idx}.png", image)

	def show_training_stats(self):
		plt.plot(self.disc_losses)
		plt.plot(self.gen_losses)
		plt.plot(self.gen_mean_losses)
		plt.legend(["Disc Loss", "Gen Loss", "Mean Gen Loss L100"])
		plt.show()
		plt.close()

	def save_models_structure_images(self, save_path:str= "."):
		if not os.path.isdir(save_path): os.mkdir(save_path)
		plot_model(self.combined_model, "combined.png", expand_nested=True, show_shapes=True)
		plot_model(self.generator, "generator.png", expand_nested=True, show_shapes=True)
		plot_model(self.discriminator, "discriminator.png", expand_nested=True, show_shapes=True)

	def clear_progress_images(self):
		if not os.path.isdir(self.progress_image_path): return

		img_file_names = os.listdir(self.progress_image_path)
		for im_file in img_file_names:
			if os.path.isfile(self.progress_image_path + "/" + im_file):
				os.remove(self.progress_image_path + "/" + im_file)

	def save_weights(self, save_directory:str= "."):
		if not os.path.isdir(save_directory): os.mkdir(save_directory)
		save_dir = os.path.join(save_directory, str(self.epoch_counter + 1))
		if not os.path.isdir(save_dir): os.mkdir(save_dir)
		self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")
		self.discriminator.save_weights(f"{save_dir}/discriminator_{self.disc_mod_name}.h5")

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