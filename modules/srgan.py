import colorama
from typing import Union
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Optimizer, Adam
from keras.models import Model
from keras.applications.vgg19 import VGG19

tf.get_logger().setLevel('ERROR')
colorama.init()

# Calculate start image size based on final image size and number of upscales
def count_upscaling_start_size(target_image_shape: tuple, num_of_upscales: int):
	upsc = (target_image_shape[0] // (2 ** num_of_upscales), target_image_shape[1] // (2 ** num_of_upscales))
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

	def __init__(self, dataset_path:str,
	             gen_mod_name: str, disc_mod_name: str,
	             training_progress_save_path:str=None,
	             generator_optimizer: Optimizer = Adam(0.0002, 0.5), discriminator_optimizer: Optimizer = Adam(0.0002, 0.5),
	             batch_size: int = 32, buffered_batches:int=20,
	             generator_weights: Union[str, None, int] = None, discriminator_weights: Union[str, None, int] = None,
	             start_episode: int = 0, load_from_checkpoint: bool = False):
		pass