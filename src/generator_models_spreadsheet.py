from keras.initializers import Initializer
from keras.layers import Layer, Dense, BatchNormalization, Reshape, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

def count_upscaling_start_size(image_shape: tuple, num_of_upscales: int):
	upsc = image_shape[0] / (2 ** num_of_upscales)
	if upsc < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
	return int(upsc)

'''
Base version 1
Kind of work with small gray images
'''
def mod_base_2upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer):
	st_s = count_upscaling_start_size(image_shape, 2)

	# (256 * st_s^2,) -> (st_s, st_s, 256)
	m = Dense(256 * st_s * st_s, kernel_initializer=kernel_initializer)(inp)
	m = BatchNormalization()(m)
	m = LeakyReLU()(m)
	m = Reshape((st_s, st_s, 256))(m)

	# (st_s, st_s, 256) -> (st_s, st_s, 256)
	m = Conv2DTranspose(256, (5, 5), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU()(m)

	# (st_s, st_s, 256) -> (2*st_s, 2*st_s, 128)
	m = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU()(m)

	# (2*st_s, 2*st_s, 128) -> (4*st_s, 4*st_s, num_ch)
	m = Conv2DTranspose(image_channels, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, activation="tanh")(m)
	return m

'''
Base version 2
Kind if works with larger color images
Tested with 1024 latent dim
'''
def mod_base_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer):
	st_s = count_upscaling_start_size(image_shape, 3)

	# (512 * st_s^2,) -> (st_s, st_s, 512)
	m = Dense(512 * st_s * st_s, kernel_initializer=kernel_initializer)(inp)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Reshape((st_s, st_s, 512))(m)

	# (st_s, st_s, 512) -> (2*st_s, 2*st_s, 256)
	m = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (2*st_s, 2*st_s, 256) -> (4*st_s, 4*st_s, 128)
	m = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 128) -> (8*st_s, 8*st_s, num_ch)
	m = Conv2DTranspose(image_channels, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, activation="tanh")(m)
	return m

'''
Base version 3
Meh results with color with medium size color images, maybe more training required or some tweaks
After 300 epochs loss starts decaying - bad!!!!!
'''
def mod_base_4upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer):
	st_s = count_upscaling_start_size(image_shape, 4)

	# (512 * st_s^2,) -> (st_s, st_s, 512)
	m = Dense(512 * st_s * st_s, kernel_initializer=kernel_initializer)(inp)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Reshape((st_s, st_s, 512))(m)

	# (st_s, st_s, 512) -> (st_s, st_s, 512)
	m = Conv2DTranspose(512, (5, 5), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (st_s, st_s, 512) -> (2*st_s, 2*st_s, 256)
	m = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (2*st_s, 2*st_s, 256) -> (4*st_s, 4*st_s, 128)
	m = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 128) -> (8*st_s, 8*st_s, 64)
	m = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 64) -> (16*st_s, 16*st_s, num_ch)
	m = Conv2DTranspose(image_channels, (5, 5), strides=(2, 2), padding="same",kernel_initializer=kernel_initializer, activation="tanh")(m)
	return m

'''
Version 4 - Extended version 2
Testing
'''
def mod_ext_4upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer):
	st_s = count_upscaling_start_size(image_shape, 4)

	# (1024 * st_s^2,) -> (st_s, st_s, 1024)
	m = Dense(1024 * st_s * st_s, kernel_initializer=kernel_initializer)(inp)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)
	m = Reshape((st_s, st_s, 1024))(m)

	# (st_s, st_s, 1024) -> (2*st_s, 2*st_s, 512)
	m = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (2*st_s, 2*st_s, 512) -> (4*st_s, 4*st_s, 256)
	m = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (4*st_s, 4*st_s, 256) -> (8*st_s, 8*st_s, 128)
	m = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization()(m)
	m = LeakyReLU(0.2)(m)

	# (8*st_s, 8*st_s, 128) -> (16*st_s, 16*st_s, num_ch)
	m = Conv2DTranspose(image_channels, (5, 5), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, activation="tanh")(m)
	return m