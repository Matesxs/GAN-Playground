from keras.layers import Layer, Conv2D, BatchNormalization, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Initializer, RandomNormal

def mod_base_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(64, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(128, (3, 3), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer=kernel_initializer)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(512, (3, 3), padding='same', strides=(1, 1), kernel_initializer=kernel_initializer)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m

def mod_ext_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m

def mod_base_6layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	m = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(inp)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m