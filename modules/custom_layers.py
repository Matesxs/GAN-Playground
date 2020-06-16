from typing import Union
from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation

def deconv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:float=0.8, conv_transpose:bool=False, leaky:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	assert filters > 0, "Invalid filter number"
	assert kernel_size > 0, "Invalid kernel size"
	assert strides > 0, "Invalid stride size"

	if conv_transpose:
		x = Conv2DTranspose(filters, (kernel_size, kernel_size), strides=(strides, strides), padding="same", kernel_initializer=kernel_initializer)(inp)
	else:
		if strides > 1:
			x = UpSampling2D(size=(strides, strides))(inp)
		else: x = inp

		x = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer)(x)

	if batch_norm: x = BatchNormalization(momentum=batch_norm)(x)
	if leaky: x = LeakyReLU(0.2)(x)
	else: x = Activation("relu")(x)
	if dropout: x = Dropout(dropout)(x)

	return x

def conv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:float=None, leaky:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	assert filters > 0, "Invalid filter number"
	assert kernel_size > 0, "Invalid kernel size"
	assert strides > 0, "Invalid stride size"

	x = Conv2D(filters, (kernel_size, kernel_size), strides=(strides, strides), padding="same", kernel_initializer=kernel_initializer)(inp)

	if batch_norm: x = BatchNormalization(momentum=batch_norm)(x)
	if leaky: x = LeakyReLU(0.2)(x)
	else: x = Activation("relu")(x)
	if dropout: x = Dropout(dropout)(x)

	return x

def identity_layer(inp:Layer, filters_number_list:Union[list, int], kernel_size:int=3, dropout:float=None, batch_norm:bool=True, leaky:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	assert kernel_size > 0, "Invalid kernel size"

	x = inp
	if isinstance(filters_number_list, list):
		for filters in filters_number_list:
			assert filters > 0, "Invalid filter number"

			x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer)

			if batch_norm: x = BatchNormalization(momentum=0.8)(x)
			if leaky: x = LeakyReLU(0.2)(x)
			else: x = Activation("relu")(x)
			if dropout: x = Dropout(dropout)(x)
	else:
		assert filters_number_list > 0, "Invalid filter number"

		x = Conv2D(filters_number_list, kernel_size=(kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer)

		if batch_norm: x = BatchNormalization(momentum=0.8)(x)
		if leaky:
			x = LeakyReLU(0.2)(x)
		else:
			x = Activation("relu")(x)
		if dropout: x = Dropout(dropout)(x)

	assert x.output_shape == inp.output_shape, "Invalid filters, input filters and output filters must be the same"
	x = Add()([x, inp])

	return x