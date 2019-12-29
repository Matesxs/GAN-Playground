import os
import matplotlib.pyplot as plt
import numpy as np
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
import random
import shutil
from tqdm import tqdm
from typing import Union

from modules.batch_maker import BatchMaker
from modules import generator_models_spreadsheet
from modules import discriminator_models_spreadsheet

tf.get_logger().setLevel('ERROR')

class DCGAN:
	def __init__(self, train_images:Union[np.ndarray, list, None, str], generator_optimizer:Optimizer=Adam(0.0002, 0.5), discriminator_optimizer:Optimizer=Adam(0.0002, 0.5), latent_dim:int=100, training_progress_save_path:str=None, progress_image_num:int=5, gen_mod_name:str= "mod_base_2upscl", disc_mod_name:str= "mod_base_4layers", discriminator_weights:str=None, generator_weights:str=None):
		self.generator_optimizer = generator_optimizer
		self.discriminator_optimizer = discriminator_optimizer

		self.disc_mod_name = disc_mod_name
		self.gen_mod_name = gen_mod_name

		self.latent_dim = latent_dim
		self.progress_image_num = progress_image_num
		self.epoch_counter = 0
		self.training_progress_save_path = training_progress_save_path

		if type(train_images) == list:
			self.train_data = np.array(train_images)
			self.data_length = self.train_data.shape[0]
		elif type(train_images) == str:
			self.train_data = [os.path.join(train_images, file) for file in os.listdir(train_images)]
			self.data_length = len(self.train_data)
		elif type(train_images) == np.ndarray:
			self.train_data = train_images
			self.data_length = self.train_data.shape[0]

		if train_images is not None:
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
		self.disc_real_losses = deque()
		self.disc_fake_losses = deque()
		self.disc_real_accs = deque()
		self.disc_fake_accs = deque()

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

	def train(self, epochs:int=200, batch_size:int=64, progress_save_interval:int=None, weights_save_interval:int=None, generator_smooth_labels:bool=False, discriminator_smooth_labels:bool=False, feed_prev_gen_batch:bool=False, disc_half_batch:bool=False, discriminator_label_noise:float=0.0):
		def noising_labels(labels: np.ndarray, noise_ammount:float=0.01):
			for idx in range(labels.shape[0]):
				if random.random() < noise_ammount:
					labels[idx] = 1 - labels[idx]
			return labels

		if self.training_progress_save_path is not None and progress_save_interval is not None and progress_save_interval <= epochs:
			if epochs%progress_save_interval != 0: raise Exception("Invalid progress save interval")
		if weights_save_interval is not None and weights_save_interval <= epochs:
			if epochs%weights_save_interval != 0: raise Exception("Invalid weights save interval")
		if self.data_length < batch_size or batch_size%2 != 0 or batch_size < 4: raise Exception("Invalid batch size")
		if self.train_data is None: raise Exception("No dataset loaded")

		num_of_batches = self.data_length // batch_size
		if disc_half_batch:
			disc_batch = batch_size // 2
		else:
			disc_batch = batch_size
		half_disc_batch = disc_batch // 2

		# Create batchmaker and start it
		batch_maker = BatchMaker(self.train_data, self.data_length, disc_batch)
		batch_maker.start()

		# Images generated in prewious batch
		last_gen_images = None

		for _ in tqdm(range(epochs), unit="ep"):
			for batch in range(num_of_batches):
				### Train Discriminator ###
				# Select batch of valid images
				imgs = batch_maker.get_batch()

				# Sample noise and generate new images
				gen_imgs = self.generator.predict(np.random.normal(0.0, 1.0, (disc_batch, self.latent_dim)))

				# Train discriminator (real as ones and fake as zeros)
				if discriminator_smooth_labels:
					disc_real_labels = np.random.uniform(0.85, 0.95, size=(disc_batch, 1))
					if feed_prev_gen_batch and last_gen_images is not None:
						disc_fake_labels = np.random.uniform(0.0, 0.1, size=(disc_batch, 1))
						tmp_imgs = np.concatenate((last_gen_images[-half_disc_batch:], gen_imgs[-half_disc_batch:]))
						last_gen_images = gen_imgs
						gen_imgs = tmp_imgs
					else:
						disc_fake_labels = np.random.uniform(0.0, 0.1, size=(disc_batch, 1))
				else:
					disc_real_labels = np.ones(shape=(disc_batch, 1))
					if feed_prev_gen_batch and last_gen_images is not None:
						disc_fake_labels = np.zeros(shape=(disc_batch, 1))
						tmp_imgs = np.concatenate((last_gen_images[-half_disc_batch:], gen_imgs[-half_disc_batch:]))
						last_gen_images = gen_imgs
						gen_imgs = tmp_imgs
					else:
						disc_fake_labels = np.zeros(shape=(disc_batch, 1))

				# Adding random noise to discriminator labels
				if discriminator_label_noise > 0:
					discriminator_label_noise /= 2
					disc_real_labels = noising_labels(disc_real_labels, discriminator_label_noise)
					disc_fake_labels = noising_labels(disc_fake_labels, discriminator_label_noise)

				self.discriminator.trainable = True
				self.discriminator.train_on_batch(imgs, disc_real_labels)
				self.discriminator.train_on_batch(gen_imgs, disc_fake_labels)
				self.discriminator.trainable = False

				### Train Generator ###
				# Train generator (wants discriminator to recognize fake images as valid)
				if generator_smooth_labels:
					gen_labels = np.random.uniform(0.7, 1.2, size=(batch_size, 1))
				else:
					gen_labels = np.ones(shape=(batch_size, 1))

				self.combined_model.train_on_batch(np.random.normal(0.0, 1.0, (batch_size, self.latent_dim)), gen_labels)

			# Generate and save statistics
			imgs = batch_maker.get_batch()
			gen_imgs = gen_imgs = self.generator.predict(np.random.normal(0.0, 1.0, (disc_batch, self.latent_dim)))
			disc_real_loss, disc_real_acc = self.discriminator.test_on_batch(imgs, np.ones(shape=(imgs.shape[0], 1)))
			disc_fake_loss, disc_fake_acc = self.discriminator.test_on_batch(gen_imgs, np.zeros(shape=(gen_imgs.shape[0], 1)))
			gen_loss = self.combined_model.train_on_batch(np.random.normal(0.0, 1.0, (batch_size, self.latent_dim)), np.ones(shape=(batch_size, 1)))

			self.gen_losses.append(gen_loss)
			self.disc_real_losses.append(disc_real_loss)
			disc_real_acc = disc_real_acc * 100
			self.disc_real_accs.append(disc_real_acc)
			self.disc_fake_losses.append(disc_fake_loss)
			disc_fake_acc = disc_fake_acc * 100
			self.disc_fake_accs.append(disc_fake_acc)

			if (self.epoch_counter + 1) % 10 == 0:
				print(f"[D-R loss: {round(float(disc_real_loss), 5)}, D-R acc: {round(disc_real_acc, 2)}%, D-F loss: {round(float(disc_fake_loss), 5)}, D-F acc: {round(disc_fake_acc, 2)}%] [G loss: {round(float(gen_loss), 5)}]")

			# Save progress
			if self.training_progress_save_path is not None and progress_save_interval is not None and (self.epoch_counter + 1) % progress_save_interval == 0:
				self.__save_imgs(self.epoch_counter)

			if weights_save_interval is not None and (self.epoch_counter + 1) % weights_save_interval == 0:
				self.save_weights()

			self.epoch_counter += 1

		# Shutdown batchmaker and wait for its exit
		batch_maker.terminate = True
		batch_maker.join()

	def __save_imgs(self, epoch):
		if not os.path.exists(self.training_progress_save_path + "/progress_images"): os.makedirs(self.training_progress_save_path + "/progress_images")
		gen_imgs = self.generator.predict(self.static_noise)

		# Rescale images 0 to 255
		gen_imgs = (0.5 * gen_imgs + 0.5) * 255

		final_image = np.zeros(shape=(self.image_shape[0] * self.progress_image_num, self.image_shape[0] * self.progress_image_num, self.image_channels)).astype(np.float32)

		cnt = 0
		for i in range(self.progress_image_num):
			for j in range(self.progress_image_num):
				if self.image_channels == 3:
					final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[0] * j:self.image_shape[0] * (j + 1), :] = gen_imgs[cnt]
				else:
					final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[0] * j:self.image_shape[0] * (j + 1), 0] = gen_imgs[cnt, :, :, 0]
				cnt += 1
		final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
		cv.imwrite(f"{self.training_progress_save_path}/progress_images/{self.epoch_counter + 1}.png", final_image)

	def generate_collage(self, img_size:tuple=(16, 9), save_path: str = ".", blur: bool = False):
		gen_imgs = self.generator.predict(np.random.normal(0.0, 1.0, size=(img_size[0] * img_size[1], self.latent_dim)))

		# Rescale images 0 to 255
		gen_imgs = (0.5 * gen_imgs + 0.5) * 255

		final_image = np.zeros(shape=(self.image_shape[0] * img_size[0], self.image_shape[0] * img_size[1],self.image_channels)).astype(np.float32)

		cnt = 0
		for i in range(img_size[0]):
			for j in range(img_size[1]):
				if self.image_channels == 3:
					final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[0] * j:self.image_shape[0] * (j + 1), :] = gen_imgs[cnt]
				else:
					final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[0] * j:self.image_shape[0] * (j + 1), 0] = gen_imgs[cnt, :, :, 0]
				cnt += 1
		final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
		cv.imwrite(f"{save_path}/collage.png", final_image)

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

	def show_training_stats(self, save_path:str=None):
		# Loss graph
		plt.subplot(2, 1, 1)
		plt.plot(self.gen_losses, label="Gen Loss")
		plt.plot(self.disc_fake_losses, label="Disc Fake Loss")
		plt.plot(self.disc_real_losses, label="Disc Real Loss")
		plt.plot(0.5 * np.add(self.disc_fake_losses, self.disc_real_losses), label="Disc Loss")
		plt.legend()

		# Acc graph
		plt.subplot(2, 1, 2)
		plt.plot(self.disc_fake_accs, label="Disc Fake Acc")
		plt.plot(self.disc_real_accs, label="Disc Real Acc")
		plt.plot(0.5 * np.add(self.disc_fake_accs, self.disc_real_accs), label="Disc Acc")
		plt.legend()

		if not save_path:
			plt.show()
		else:
			plt.savefig(f"{save_path}/training_stats.png")
		plt.close()

	def save_models_structure_images(self, save_path:str=None):
		if save_path is None: save_path = self.training_progress_save_path + "/model_structures"
		if not os.path.exists(save_path): os.makedirs(save_path)
		plot_model(self.combined_model, os.path.join(save_path,"combined.png"), expand_nested=True, show_shapes=True)
		plot_model(self.generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)
		plot_model(self.discriminator, os.path.join(save_path, "discriminator.png"), expand_nested=True, show_shapes=True)

	def clear_training_progress_folder(self):
		if not os.path.exists(self.training_progress_save_path): return
		shutil.rmtree(self.training_progress_save_path, ignore_errors=True)

	def save_weights(self):
		save_dir = self.training_progress_save_path + "/weights/" + str(self.epoch_counter + 1)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")
		self.discriminator.save_weights(f"{save_dir}/discriminator_{self.disc_mod_name}.h5")

	def make_progress_gif(self, save_path:str=None, duration:int=120):
		if not os.path.exists(self.training_progress_save_path + "/progress_images"): return
		if not save_path: save_path = self.training_progress_save_path
		if not os.path.exists(save_path): os.makedirs(save_path)

		frames = []
		img_file_names = os.listdir(self.training_progress_save_path + "/progress_images")

		for im_file in img_file_names:
			if os.path.isfile(self.training_progress_save_path + "/progress_images/" + im_file):
				frames.append(Image.open(self.training_progress_save_path + "/progress_images/" + im_file))

		if len(frames) > 2:
			frames[0].save(f"{save_path}/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=0)