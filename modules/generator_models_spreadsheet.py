from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Dense, BatchNormalization, Reshape, Conv2DTranspose, UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU

def count_upscaling_start_size(image_shape: tuple, num_of_upscales: int):
	upsc = image_shape[0] / (2 ** num_of_upscales)
	if upsc < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
	return int(upsc)

def mod_base_2upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 2)

	# (256 * st_s^2,) -> (st_s, st_s, 256)
	m = Dense(256 * st_s * st_s, kernel_initializer=kernel_initializer)(inp)
	m = Reshape((st_s, st_s, 256))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 256) -> (st_s, st_s, 256)
	m = Conv2DTranspose(256, (4, 4), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU()(m)

	# (st_s, st_s, 256) -> (2*st_s, 2*st_s, 128)
	m = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU()(m)

	# (2*st_s, 2*st_s, 128) -> (4*st_s, 4*st_s, num_ch)
	m = Conv2DTranspose(image_channels, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, activation="tanh")(m)
	return m

def mod_mod_2upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 2)

	# (256 * st_s^2,) -> (st_s, st_s, 256)
	m = Dense(256 * st_s * st_s, kernel_initializer=kernel_initializer, activation="relu")(inp)
	m = Reshape((st_s, st_s, 256))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 256) -> (st_s, st_s, 256)
	m = Conv2D(256, (4, 4), padding="same", kernel_initializer=kernel_initializer, activation="relu")(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 256) -> (2*st_s, 2*st_s, 128)
	m = UpSampling2D()(m)
	m = Conv2D(128, (4, 4), padding="same", kernel_initializer=kernel_initializer, activation="relu")(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (2*st_s, 2*st_s, 128) -> (4*st_s, 4*st_s, num_ch)
	m = UpSampling2D()(m)
	m = Conv2D(image_channels, (4, 4), padding="same", kernel_initializer=kernel_initializer, activation="tanh")(m)
	return m

def mod_base_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 3)

	# (128 * st_s^2,) -> (st_s, st_s, 128)
	m = Dense(128 * st_s * st_s, kernel_initializer=kernel_initializer, activation="relu")(inp)
	m = Reshape((st_s, st_s, 128))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 128) -> (2*st_s, 2*st_s, 1024)
	m = UpSampling2D()(m)
	m = Conv2D(1024, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
	m = UpSampling2D()(m)
	m = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
	m = UpSampling2D()(m)
	m = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 128)
	m = Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, 64)
	m = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (8*st_s, 8*st_s, 64) -> (8*st_s, 8*st_s, 32)
	m = Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (8*st_s, 8*st_s, 32) -> (8*st_s, 8*st_s, num_ch)
	m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
	return m

def mod_min_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 3)

	# (128 * st_s^2,) -> (st_s, st_s, 128)
	m = Dense(128 * st_s * st_s, kernel_initializer=kernel_initializer, activation="relu")(inp)
	m = Reshape((st_s, st_s, 128))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 128) -> (2*st_s, 2*st_s, 1024)
	m = UpSampling2D()(m)
	m = Conv2D(1024, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
	m = UpSampling2D()(m)
	m = Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
	m = UpSampling2D()(m)
	m = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 128)
	m = Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, num_ch)
	m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
	return m