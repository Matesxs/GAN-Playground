import os
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import Adam, Optimizer
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Reshape, Dense, Flatten, BatchNormalization, Activation, Conv2DTranspose, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.utils import plot_model
import tensorflow as tf
from PIL import Image
import time
from tqdm import tqdm
from typing import Union

tf.get_logger().setLevel('ERROR')

class BIGAN:
	def __init__(self, train_images:Union[np.ndarray, list, None], optimizer:Optimizer=Adam(0.0002, 0.5), latent_dim:int=100, ex:int=5, gen_v:int=1, disc_v:int=1):
		self.optimizer = optimizer
		self.latent_dim = latent_dim
		self.ex = ex

		if type(train_images) == list:
			self.train_data = np.array(train_images)
		elif train_images is None:
			(x_train, y_train), (x_test, y_test) = cifar10.load_data()
			# Selecting cats :)
			x_train = x_train[np.where(y_train == 3)[0]]
			x_test = x_test[np.where(y_test == 3)[0]]
			x_train = np.concatenate((x_train, x_test))
			self.train_data = x_train
		else:
			self.train_data = train_images

		self.image_shape = self.train_data[0].shape
		self.image_rows = self.image_shape[0]
		self.image_cols = self.image_shape[1]
		self.image_channels = self.image_shape[2]

		self.static_noise = np.random.normal(size=(self.ex*self.ex, self.latent_dim))
		self.conv_kerner_initializer = RandomNormal(stddev=0.02)

		# Build discriminator
		self.discriminator = self.build_discriminator(disc_v)
		self.discriminator.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=["binary_accuracy"])

		# Build generator
		self.generator = self.build_generator(gen_v)

		# Generator takes noise and generates images
		z = Input(shape=(self.latent_dim,))
		img = self.generator(z)

		# For combined model we will only train generator
		self.discriminator.trainable = False

		# Discriminator takes images and determinates validity
		valid = self.discriminator(img)

		# Combine models
		# Train generator to fool discriminator
		self.combined_model = Model(z, valid)
		self.combined_model.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=['binary_accuracy'])

		# Statistics
		self.gen_losses = []
		self.disc_losses = []

	def build_generator(self, version:int=1):
		model = Sequential()

		if version == 1:
			# (16384,) -> (8, 8, 256)
			model.add(Dense(256 * 8 * 8, input_shape=(self.latent_dim,), kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU())
			model.add(Reshape((8, 8, 256)))

			# (8, 8, 256) -> (8, 8, 256)
			model.add(Conv2DTranspose(256, (5, 5), strides=(1, 1), padding="same", kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU())

			# (8, 8, 256) -> (16, 16, 128)
			model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.conv_kerner_initializer))
			model.add(BatchNormalization())
			model.add(LeakyReLU())

			# (16, 16, 128) -> (32, 32, num_ch)
			model.add(Conv2DTranspose(self.image_channels, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.conv_kerner_initializer))
			model.add(Activation("tanh"))
		else:
			# (2048,) -> (2, 2, 512)
			model.add(Dense(2 * 2 * 512, input_shape=(self.latent_dim,), kernel_initializer=self.conv_kerner_initializer))
			model.add(Reshape((2, 2, 512)))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (2, 2, 512) -> (4, 4, 256)
			model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (4, 4, 256) -> (8, 8, 128)
			model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (8, 8, 128) -> (16, 16, 64)
			model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"))
			model.add(BatchNormalization())
			model.add(LeakyReLU(0.2))

			# (16, 16, 64) -> (32, 32, num_ch)
			model.add(Conv2DTranspose(self.image_channels, (5, 5), strides=(2, 2), padding="same", activation="tanh"))

		print("\nGenerator Sumary:")
		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self, version:int=1):
		model = Sequential()

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
		else:
			model.add(Conv2D(32, 3, padding='same', strides=2, input_shape=self.image_shape, kernel_initializer=self.conv_kerner_initializer))
			model.add(LeakyReLU(0.2))
			model.add(Dropout(0.3))

			model.add(Conv2D(64, 3, padding='same', strides=1))
			model.add(LeakyReLU(0.2))
			model.add(Dropout(0.3))

			model.add(Conv2D(128, 3, padding='same', strides=2))
			model.add(LeakyReLU(0.2))
			model.add(Dropout(0.3))

			model.add(Conv2D(256, 3, padding='same', strides=1))
			model.add(LeakyReLU(0.2))
			model.add(Dropout(0.3))

			model.add(Flatten())

		model.add(Dense(1, activation="sigmoid"))

		print("\nDiscriminator Sumary:")
		model.summary()

		img = Input(shape=self.image_shape)
		validity = model(img)

		return Model(img, validity)

	def train(self, epochs:int=800, batch_size:int=64, save_interval:int=50, smooth:float=0.1):
		# Clear statistics
		self.gen_losses = []
		self.disc_losses = []

		# Scale -1 to 1
		self.train_data = self.train_data / 127.5 - 1.0

		# Validity arrays
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		if not os.path.isdir("prog_images"):
			os.mkdir("prog_images")

		g_loss, d_loss = None, None

		s_time = time.time()
		for epoch in range(epochs):
			for batch in tqdm(range(self.train_data.shape[0] // batch_size), unit="batch"):
				### Train Discriminator ###
				# Select batch of valid images
				idx = np.random.randint(0, self.train_data.shape[0], batch_size)
				imgs = self.train_data[idx]

				# Sample noise and generate new images
				noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
				gen_imgs = self.generator.predict(noise)

				# Train discriminator (real as ones and fake as zeros)
				self.discriminator.trainable = True
				d_loss_real = self.discriminator.train_on_batch(imgs, valid * (1 - smooth))
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
				self.discriminator.trainable = False
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				### Train Generator ###
				# Train generator (wants discriminator to recognize fake images as valid)
				g_loss = self.combined_model.train_on_batch(noise, valid)

			# Save statistics
			self.gen_losses.append(g_loss)
			self.disc_losses.append(d_loss)

			# Save progress
			if (epoch + 1) % save_interval == 0:
				print(f"{epoch + 1} - [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss[0]}, acc.: {100 * g_loss[1]}] - Elapsed: {round((time.time() - s_time) / 60, 1)}min")
				self.__save_imgs(epoch)

	def __save_imgs(self, epoch):
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

		fig.savefig(f"prog_images/{epoch + 1}.png")
		plt.close()

	def show_sample_of_dataset(self):
		if self.train_data is None: return

		fig, axs = plt.subplots(self.ex, self.ex)
		cnt = 0
		for i in range(self.ex):
			for j in range(self.ex):
				if self.image_channels == 3:
					axs[i, j].imshow(self.train_data[np.random.randint(0, self.train_data.shape[0], size=1)][0])
				else:
					axs[i, j].imshow(self.train_data[np.random.randint(0, self.train_data.shape[0], size=1), :, :, 0][0], cmap="gray")
				axs[i, j].axis('off')
				cnt += 1
		plt.show()
		plt.close()

	def show_training_stats(self):
		disc_losses = np.array(self.disc_losses)
		gen_losses = np.array(self.gen_losses)

		plt.plot(disc_losses[:, 0])
		plt.plot(gen_losses[:, 0])
		plt.plot(gen_losses[:, 1])
		plt.legend(["Disc Loss", "Gen Loss", "Gen Acc"])
		plt.show()
		plt.close()

	def plot_models(self):
		plot_model(self.combined_model, "combined.png", expand_nested=True)
		plot_model(self.generator, "generator.png", expand_nested=True)
		plot_model(self.discriminator, "discriminator.png", expand_nested=True)

	def clear_output_folder(self):
		if not os.path.isdir("prog_images"): return

		img_file_names = os.listdir("prog_images")
		for im_file in img_file_names:
			if os.path.isfile("prog_images/" + im_file):
				os.remove("prog_images/" + im_file)

	def save_models(self):
		self.generator.save("generator.h5")
		self.discriminator.save("discriminator.h5")

	def save_models_prompt(self):
		while True:
			ans = input("Do you want to save models? (y/n)\n")
			if ans == "y":
				self.save_models()
				break
			elif ans == "n":
				break

	def make_gif(self):
		frames = []
		img_file_names = os.listdir("prog_images")

		for im_file in img_file_names:
			if os.path.isfile("prog_images/" + im_file):
				im = Image.open("prog_images/" + im_file)
				frames.append(im)

		if len(frames) > 2:
			frames[0].save("prog_images/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, duration=120, loop=0)

if __name__ == '__main__':
	gan = BIGAN(None, latent_dim=100, gen_v=2, disc_v=2)
	# gan.plot_models()
	# gan.show_sample_of_dataset()
	gan.clear_output_folder()
	gan.train(500, 16, 10)
	gan.make_gif()
	gan.show_training_stats()
	gan.save_models_prompt()