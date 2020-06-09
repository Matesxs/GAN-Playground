from keras import layers
from keras.initializers import Initializer, RandomNormal

def deconvolutional_block(X, k, filters, s=2, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	"""
	Arguments:
	X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	k -- integer, specifying the shape of the middle CONV's window for the main path
	filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	stage -- integer, used to name the layers, depending on their position in the network
	block -- string/character, used to name the layers, depending on their position in the network
	s -- Integer, specifying the stride to be used

	Returns:
	X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
	"""

	# Retrieve Filters
	F1, F2, F3 = filters

	# Save the input value
	X_shortcut = X

	##### MAIN PATH #####
	# First component of main path
	X = layers.Conv2DTranspose(F1, (1, 1), strides=(s, s), kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)
	X = layers.Activation('relu')(X)

	# Second component of main path (≈3 lines)
	X = layers.Conv2DTranspose(filters=F2, kernel_size=(k, k), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)
	X = layers.Activation('relu')(X)

	# Third component of main path (≈2 lines)
	X = layers.Conv2DTranspose(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)

	##### SHORTCUT PATH #### (≈2 lines)
	X_shortcut = layers.Conv2DTranspose(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=kernel_initializer)(X_shortcut)
	X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
	X = layers.Add()([X, X_shortcut])
	X = layers.Activation('relu')(X)

	return X


def convolutional_block(X, k, filters, s=2, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	"""
	Arguments:
	X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	k -- integer, specifying the shape of the middle CONV's window for the main path
	filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	stage -- integer, used to name the layers, depending on their position in the network
	block -- string/character, used to name the layers, depending on their position in the network
	s -- Integer, specifying the stride to be used

	Returns:
	X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
	"""

	# Retrieve Filters
	F1, F2, F3 = filters

	# Save the input value
	X_shortcut = X

	##### MAIN PATH #####
	# First component of main path
	X = layers.Conv2D(F1, (1, 1), strides=(s, s), kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)
	X = layers.LeakyReLU()(X)

	# Second component of main path (≈3 lines)
	X = layers.Conv2D(filters=F2, kernel_size=(k, k), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)
	X = layers.LeakyReLU()(X)

	# Third component of main path (≈2 lines)
	X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)

	##### SHORTCUT PATH #### (≈2 lines)
	X_shortcut = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=kernel_initializer)(X_shortcut)
	X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
	X = layers.Add()([X, X_shortcut])
	X = layers.LeakyReLU()(X)

	return X

def identity_block(X, k, filters, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
	# Retrieve Filters
	F1, F2, F3 = filters

	# Save the input value. You'll need this later to add back to the main path.
	X_shortcut = X

	# First component of main path
	X = layers.Conv2DTranspose(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)

	# Second component of main path (≈3 lines)
	X = layers.Conv2DTranspose(filters=F2, kernel_size=(k, k), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)

	# Third component of main path (≈2 lines)
	X = layers.Conv2DTranspose(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=kernel_initializer)(X)
	X = layers.BatchNormalization(axis=3)(X)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
	X = layers.Add()([X, X_shortcut])

	return X
