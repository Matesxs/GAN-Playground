from keras.layers import Layer, Conv2D, BatchNormalization, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Initializer

def mod_base_4layers(inp:Layer, kernel_initializer:Initializer):
	m = Conv2D(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(128, (5, 5), strides=(2, 2), padding="same")(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(256, (5, 5), strides=(2, 2), padding="same")(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(512, (5, 5), strides=(2, 2), padding="same")(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Flatten()(m)
	return m

def mod_base_5layers(inp:Layer, kernel_initializer:Initializer):
	m = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(64, (5, 5), padding='same', strides=(2, 2))(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(128, (5, 5), padding='same', strides=(2, 2))(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(256, (5, 5), padding='same', strides=(2, 2))(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Conv2D(512, (5, 5), padding='same', strides=(2, 2))(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	m = Flatten()(m)
	return m

def mod_min_4layers(inp:Layer, kernel_initializer:Initializer):
	m = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(64, (5, 5), padding='same', strides=(2, 2))(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(128, (5, 5), padding='same', strides=(2, 2))(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Conv2D(256, (5, 5), padding='same', strides=(2, 2))(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Dropout(0.25)(m)

	m = Flatten()(m)
	return m