from keras.layers import Layer, Conv2D, BatchNormalization, Flatten, Dropout, ZeroPadding2D, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Initializer, RandomNormal
from keras.constraints import Constraint
import keras.backend as K

# Custom constrain for clipping values on conv2d layers
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

def mod_base_4layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02), constrain:Constraint=None):
	m = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Flatten()(m)
	return m

def mod_base_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02), constrain:Constraint=None):
	m = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(64, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Conv2D(128, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Conv2D(256, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Conv2D(512, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Flatten()(m)
	return m

def mod_ext_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02), constrain:Constraint=None):
	m = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(inp)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(64, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = ZeroPadding2D(padding=((0, 1), (0, 1)))(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Conv2D(128, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(512, (3, 3), padding='same', strides=(1, 1), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m

def mod_min_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02), constrain:Constraint=None):
	m = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(inp)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(64, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = ZeroPadding2D(padding=((0, 1), (0, 1)))(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Conv2D(128, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)
	m = BatchNormalization(momentum=0.8)(m)

	m = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m

def mod_base_8layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02), constrain:Constraint=None):
	m = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)

	m = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)

	m = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)

	m = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)

	m = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)

	m = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)

	m = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer, kernel_constraint=constrain)(m)
	m = LeakyReLU(0.2)(m)
	m = BatchNormalization(momentum=0.8)

	m = Dense(1024)(m)
	return m