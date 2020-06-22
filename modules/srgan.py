import colorama
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
import cv2 as cv
import json

from modules.models import upscaling_generator_models_spreadsheet, discriminator_models_spreadsheet
from modules.custom_tensorboard import TensorBoardCustom
from modules.batch_maker import BatchMaker

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
		# Make trainable as False
		for l in vgg19.layers:
			l.trainable = False
		model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
		model.trainable = False

		return K.mean(K.square(model(y_true) - model(y_pred)))

class SRGAN:
	AGREGATE_STAT_INTERVAL = 1  # Interval of saving data
	GRADIENT_CHECK_INTERVAL = 10  # Interval of checking norm gradient value of combined model
	CHECKPOINT_SAVE_INTERVAL = 1  # Interval of saving checkpoint

	def __init__(self, dataset_path:str, num_of_upscales:int,
	             gen_mod_name: str, disc_mod_name: str,
	             training_progress_save_path:str=None,
	             generator_optimizer: Optimizer = Adam(0.0002, 0.5), discriminator_optimizer: Optimizer = Adam(0.0002, 0.5),
	             batch_size: int = 32, buffered_batches:int=20,
	             generator_weights: Union[str, None, int] = None, discriminator_weights: Union[str, None, int] = None,
	             start_episode: int = 0, load_from_checkpoint: bool = False):

		self.disc_mod_name = disc_mod_name
		self.gen_mod_name = gen_mod_name
		self.num_of_upscales = num_of_upscales

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
		self.discriminator.compile(loss=self.loss_object.vgg_loss, optimizer=generator_optimizer)
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
		self.combined_generator_model = Model(small_image_input, [gen_images, valid], name="srgan_model")
		self.combined_generator_model.compile(loss=[self.loss_object.vgg_loss, "binary_crossentropy"],
		                                      loss_weights=[1., 1e-3],
		                                      optimizer=generator_optimizer)

		# Load checkpoint
		if load_from_checkpoint: self.load_checkpoint()

		# Load weights from param and override checkpoint weights
		if generator_weights: self.generator.load_weights(generator_weights)
		if discriminator_weights: self.discriminator.load_weights(discriminator_weights)

	# Check if datasets have consistent shapes
	def validate_dataset(self):
		for im_path in self.train_data:
			im_shape = cv.imread(im_path).shape
			if im_shape != self.target_image_shape:
				raise Exception("Inconsistent datasets")
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

	def load_checkpoint(self):
		checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
		if not os.path.exists(os.path.join(checkpoint_base_path, "checkpoint_data.json")): return

		with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "rb") as f:
			data = json.load(f)

			if data:
				self.epoch_counter = int(data["episode"])
				self.generator.load_weights(data["gen_path"])
				self.discriminator.load_weights(data["disc_path"])

	def save_checkpoint(self):
		checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
		if not os.path.exists(checkpoint_base_path): os.makedirs(checkpoint_base_path)

		self.generator.save_weights(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5")
		self.discriminator.save_weights(f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5")

		data = {
			"episode": self.epoch_counter,
			"gen_path": f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5",
			"disc_path": f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5"
		}

		with open(os.path.join(checkpoint_base_path, "checkpoint_data.json", encoding='utf-8'), "w") as f:
			json.dump(data, f)