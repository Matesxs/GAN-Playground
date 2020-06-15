import keras
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend as K

class TensorBoardCustom(Callback):
	def __init__(self, log_dir):
		super().__init__()
		self.step = 0
		self.log_dir = log_dir
		self.writer = None

	def __del__(self):
		try:
			if self.writer:
				self.writer.close()
		except:
			pass

	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(self.step, **logs)

	def init_writer_check(self):
		if not self.writer:
			self.writer = tf.summary.create_file_writer(self.log_dir)

	def log_weights(self, model:keras.Model):
		self.init_writer_check()

		with self.writer.as_default():
			for layer in model.layers:
				for weight in layer.weights:
					weight_name = weight.name.replace(':', '_')
					weight = K.get_value(weight)
					tf.summary.histogram(weight_name, weight, step=self.step)
		self.writer.flush()

	def update_stats(self, step, **stats):
		self._write_logs(stats, step)

	# More or less the same writer as in Keras' Tensorboard callback
	# Physically writes to the log files
	def _write_logs(self, logs, index):
		self.init_writer_check()

		with self.writer.as_default():
			for name, value in logs.items():
				if name in ['batch', 'size']:
					continue

				if "actions" in name:
					for action in value.keys():
						tf.summary.scalar(f"A_{action}", value[action], step=index)
				else:
					tf.summary.scalar(name, value, step=index)
		self.writer.flush()