from typing import Union
import tensorflow as tf
from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Conv2D, UpSampling2D, BatchNormalization, Dropout, Add, PReLU, Lambda, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation

from ..keras_extensions.spectral_normalization import ConvSN2D
from ..keras_extensions.subpixel_conv import SubpixelConv2D

def deconv_layer(inp:Union[Layer, tf.Tensor], filters:int, kernel_size:int=3, upscale_multiplier:int=2, dropout:float=None, batch_norm:Union[float, None]=None, use_subpixel_conv2d:bool=False, act:Union[str, None]= "leaky", use_bias:bool=True, use_sn:bool=False, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert upscale_multiplier > 0, "Invalid upscale_multiplier, value must be >= 1"

  conv_layer = Conv2D(filters, kernel_size, strides=1, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None) if not use_sn else ConvSN2D(filters, kernel_size, strides=1, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)

  if upscale_multiplier > 1:
    if use_subpixel_conv2d:
      x = conv_layer(inp)
      x = SubpixelConv2D(upscale_multiplier)(x)
    else:
      x = conv_layer(inp)
      x = UpSampling2D(size=upscale_multiplier)(x)
  else:
    x = conv_layer(inp)

  if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
  if act == "leaky": x = LeakyReLU(0.2)(x)
  elif act == "prelu": x = PReLU(shared_axes=[1, 2])(x)
  elif act == "relu": x = Activation("relu")(x)
  if dropout: x = Dropout(dropout)(x)

  return x

def conv_layer(inp:Union[Layer, tf.Tensor], filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:Union[float, None]=None, act:Union[str, None]="leaky", use_bias:bool=True, use_sn:bool=False, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  conv_layer = Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None) if not use_sn else ConvSN2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)

  x = conv_layer(inp)

  if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
  if act == "leaky": x = LeakyReLU(0.2)(x)
  elif act == "prelu": x = PReLU(shared_axes=[1, 2])(x)
  elif act == "relu": x = Activation("relu")(x)
  if dropout: x = Dropout(dropout)(x)

  return x

def res_block(inp:Union[Layer, tf.Tensor], filters:int, kernel_size:int=3, strides:int=2, batch_norm:Union[float, None]=0.5, use_bias:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  gen = inp

  model = Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)

  if batch_norm:
    model = BatchNormalization(momentum=batch_norm, axis=-1)(model)

  model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
  model = Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(model)

  if batch_norm:
    model = BatchNormalization(momentum=batch_norm, axis=-1)(model)

  model = Add()(inputs=[gen, model])

  return model

def RRDB1(inp:Union[Layer, tf.Tensor], filters:int=64, kernel_size:int=3, use_bias:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  def dense_block(inp):
    x1 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(inp)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([inp, x1])

    x2 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([inp, x1, x2])

    x3 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x2)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([inp, x1, x2, x3])

    x4 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x3)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([inp, x1, x2, x3, x4])

    x5 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x4)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    x = Add()([x5, inp])
    return x

  x = dense_block(inp)
  x = dense_block(x)
  x = dense_block(x)
  x = Lambda(lambda x: x * 0.2)(x)
  out = Add()([x, inp])
  return out

def RRDB2(inp:Union[Layer, tf.Tensor], filters:int=64, kernel_size:int=3, use_bias:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02),
          conv_layers_in_dense_block:int=4, num_of_RDBs:int=3):
  def _dense_block(input_layer):
    """
    Implementation of the (Residual) Dense Block as in the paper
    Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).
    Residuals are incorporated in the RRDB.
    """

    x = input_layer
    for _ in range(conv_layers_in_dense_block):
      F_dc = Conv2D(
        filters,
        kernel_size=kernel_size,
        padding='same',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer
      )(x)
      F_dc = Activation('relu')(F_dc)
      x = Concatenate(axis=3)([x, F_dc])

    x = Conv2D(
      filters,
      kernel_size=3,
      padding='same',
      use_bias=use_bias,
      kernel_initializer=kernel_initializer
    )(x)
    return x

  x = inp

  for d in range(num_of_RDBs):
    LFF = _dense_block(x)
    LFF_beta = Lambda(lambda x: x * 0.2)(LFF)
    x = Add()([x, LFF_beta])
  x = Lambda(lambda x: x * 0.2)(x)
  x = Add()([inp, x])
  return x