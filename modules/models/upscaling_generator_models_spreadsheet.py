from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, BatchNormalization, Conv2D, PReLU, Add

from modules.models.custom_layers import deconv_layer, res_block

def mod_srgan_base(inp:Layer, start_image_shape:tuple, num_of_upscales:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):

	m = Conv2D(filters=64, kernel_size=9, strides=1, padding="same", kernel_initializer=kernel_initializer, input_shape=start_image_shape)(inp)
	m = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(m)

	skip = m

	for _ in range(16):
		m = res_block(m, 64, 3, 1, kernel_initializer=kernel_initializer)

	m = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_initializer=kernel_initializer)(m)
	m = BatchNormalization(momentum=0.5)(m)
	m = Add([skip, m])

	for _ in range(num_of_upscales):
		m = deconv_layer(m, 256, kernel_size=3, strides=2, leaky=True, kernel_initializer=kernel_initializer)

	m = Conv2D(filters=start_image_shape[2], kernel_size=9, strides=1, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
	return m