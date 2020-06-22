from keras.layers import Conv2D, BatchNormalization, PReLU, add, UpSampling2D, LeakyReLU, Layer, Activation, Conv2DTranspose, Dropout
from keras.initializers import Initializer, RandomNormal

def res_block(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	assert filters > 0, "Invalid filter number"
	assert kernel_size > 0, "Invalid kernel size"
	assert strides > 0, "Invalid stride size"

	gen = inp

	model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer)(inp)
	model = BatchNormalization(momentum=0.5)(model)

	model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
	model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer)(model)
	model = BatchNormalization(momentum=0.5)(model)

	model = add([gen, model])

	return model

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

def deconv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:float=0.8, conv_transpose:bool=False, leaky:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	assert filters > 0, "Invalid filter number"
	assert kernel_size > 0, "Invalid kernel size"
	assert strides > 0, "Invalid stride size"

	if conv_transpose:
		x = Conv2DTranspose(filters, (kernel_size, kernel_size), strides=(strides, strides), padding="same", kernel_initializer=kernel_initializer)(inp)
	else:
		x = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer)(inp)
		if strides > 1:
			x = UpSampling2D(size=(strides, strides))(x)
		else: x = inp

	if batch_norm: x = BatchNormalization(momentum=batch_norm)(x)
	if leaky: x = LeakyReLU(0.2)(x)
	else: x = Activation("relu")(x)
	if dropout: x = Dropout(dropout)(x)

	return x