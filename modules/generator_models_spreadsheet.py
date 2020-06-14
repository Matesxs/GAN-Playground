from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Dense, BatchNormalization, Reshape, UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU

def count_upscaling_start_size(image_shape: tuple, num_of_upscales: int):
	upsc = (image_shape[0] // (2 ** num_of_upscales), image_shape[1] // (2 ** num_of_upscales))
	if upsc[0] < 1 or upsc[1] < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
	return upsc

def mod_base_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 3)

	# (64 * st_s^2,) -> (st_s, st_s, 64)
	m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation="relu")(inp)
	m = Reshape((st_s[0], st_s[1], 64))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
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

def mod_base_3upscl_test(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 3)

	# (64 * st_s^2,) -> (st_s, st_s, 64)
	m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	m = Reshape((st_s[0], st_s[1], 64))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
	m = UpSampling2D()(m)
	m = Conv2D(1024, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
	m = UpSampling2D()(m)
	m = Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
	m = UpSampling2D()(m)
	m = Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 128)
	m = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, 64)
	m = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 64) -> (8*st_s, 8*st_s, 32)
	m = Conv2D(32, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 32) -> (8*st_s, 8*st_s, num_ch)
	m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
	return m

def mod_ext_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 3)

	# (128 * st_s^2,) -> (st_s, st_s, 128)
	m = Dense(128 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	m = Reshape((st_s[0], st_s[1], 128))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 128) -> (2*st_s, 2*st_s, 512)
	m = UpSampling2D()(m)
	m = Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (2*st_s, 2*st_s, 512) -> (2*st_s, 2*st_s, 512)
	m = Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (2*st_s, 2*st_s, 512) -> (4*st_s, 4*st_s, 256)
	m = UpSampling2D()(m)
	m = Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 256) -> (4*st_s, 4*st_s, 256)
	m = Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 256) -> (8*st_s, 8*st_s, 128)
	m = UpSampling2D()(m)
	m = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, 128)
	m = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, 64)
	m = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 64) -> (8*st_s, 8*st_s, 64)
	m = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 64) -> (8*st_s, 8*st_s, image_channels)
	m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
	return m

def mod_ext_3upscl_test(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	st_s = count_upscaling_start_size(image_shape, 3)

	# (64 * st_s^2,) -> (st_s, st_s, 64)
	m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer)(inp)
	m = LeakyReLU(0.2)(m)
	m = Reshape((st_s[0], st_s[1], 64))(m)
	m = BatchNormalization(momentum=0.8)(m)

	# (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
	m = UpSampling2D()(m)
	m = Conv2D(1024, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
	m = UpSampling2D()(m)
	m = Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 512) -> (4*st_s, 4*st_s, 512)
	m = Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
	m = UpSampling2D()(m)
	m = Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 256)
	m = Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 128)
	m = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, 64)
	m = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.8)(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 64) -> (8*st_s, 8*st_s, image_channels)
	m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
	return m