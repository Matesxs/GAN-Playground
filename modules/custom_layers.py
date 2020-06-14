from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation

def deconv_layer(inp:Layer, filters:int, kernel_size:int=3, dropout:float=None, batch_norm:bool=True, conv_transpose:bool=False, leaky:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	if conv_transpose:
		x = Conv2DTranspose(filters, (kernel_size, kernel_size), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(inp)
	else:
		x = UpSampling2D()(inp)
		x = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer)(x)

	if batch_norm: x = BatchNormalization(momentum=0.8)(x)
	if leaky: x = LeakyReLU(0.2)(x)
	else: x = Activation("relu")(x)
	if dropout: x = Dropout(dropout)(x)

	return x

def conv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:bool=True, leaky:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	x = Conv2D(filters, (kernel_size, kernel_size), strides=(strides, strides), padding="same", kernel_initializer=kernel_initializer)(inp)

	if batch_norm: x = BatchNormalization(momentum=0.8)(x)
	if leaky: x = LeakyReLU(0.2)(x)
	else: x = Activation("relu")(x)
	if dropout: x = Dropout(dropout)(x)

	return x

def identity_layer(inp:Layer, filters_number_list:list, kernel_size:int=3, dropout:float=None, batch_norm:bool=True, leaky:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	x = inp
	for filters in filters_number_list:
		x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer)

		if batch_norm: x = BatchNormalization(momentum=0.8)(x)
		if leaky: x = LeakyReLU(0.2)(x)
		else: x = Activation("relu")(x)
		if dropout: x = Dropout(dropout)(x)

	x = Add()([x, inp])

	return x