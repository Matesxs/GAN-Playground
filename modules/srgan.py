import colorama
from typing import Union
import tensorflow as tf
from keras.optimizers import Optimizer, Adam

tf.get_logger().setLevel('ERROR')
colorama.init()

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