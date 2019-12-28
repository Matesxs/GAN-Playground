from keras.layers import Layer, Conv2D, BatchNormalization, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Initializer, RandomNormal

def mod_base_4layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(512, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Flatten()(m)
	return m

def mod_extD_4layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.5)(m)

	m = Conv2D(64, (3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.5)(m)

	m = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.5)(m)

	m = Conv2D(256, (3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.5)(m)

	m = Flatten()(m)
	return m

def mod_base_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(128, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(256, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(512, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Flatten()(m)
	return m

def mod_ext_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(128, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(256, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(512, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(1024, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Flatten()(m)
	return m

def mod_extD_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	# m = Dropout(0.25)(m)

	m = Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(128, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(512, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m

def mod_base_6layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(128, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(256, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(512, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(1024, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Flatten()(m)
	return m

def mod_extD_6layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(128, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(512, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(1024, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m