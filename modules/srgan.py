import colorama
import os
from typing import Union
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Optimizer, Adam
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.initializers import RandomNormal
import cv2 as cv

from modules.custom_tensorboard import TensorBoardCustom

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

	# Check if datasets have consistent shapes
	def validate_dataset(self):
		for im_path in self.train_data:
			im_shape = cv.imread(im_path).shape
			if im_shape != self.target_image_shape:
				raise Exception("Inconsistent datasets")
		print("Dataset valid")
