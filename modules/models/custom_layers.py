from typing import Union
import tensorflow as tf
from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Conv2D, UpSampling2D, BatchNormalization, Dropout, Add, PReLU, Lambda, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation

from modules.keras_extensions.spectral_normalization import ConvSN2D

subpixel_index = 0
def SubpixelConv2D(scale=2):
  global subpixel_index

  def subpixel_shape(input_shape):
    dims = [input_shape[0],
            None if input_shape[1] is None else input_shape[1] * scale,
            None if input_shape[2] is None else input_shape[2] * scale,
            int(input_shape[3] / (scale ** 2))]
    output_shape = tuple(dims)
    return output_shape

  def subpixel(x):
    return tf.nn.depth_to_space(x, scale)

  subpixel_index += 1
  return Lambda(subpixel, output_shape=subpixel_shape, name=f"subpixel_conv2d_{subpixel_index}")

def deconv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:Union[float, None]=None, use_subpixel_conv2d:bool=False, act:Union[str, None]="leaky", upsample_first:bool=True, use_bias:bool=True, use_sn:bool=False, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  if upsample_first:
    if strides > 1:
      if use_subpixel_conv2d:
        x = SubpixelConv2D(strides)(inp)
      else:
        x = UpSampling2D(size=strides)(inp)
    else: x = inp

    if use_sn:
      x = ConvSN2D(filters, kernel_size, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(x)
    else:
      x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(x)
  else:
    if use_sn:
      x = ConvSN2D(filters, kernel_size, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)
    else:
      x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)

    if strides > 1:
      if use_subpixel_conv2d:
        x = SubpixelConv2D(strides)(x)
      else:
        x = UpSampling2D(size=strides)(x)
    else: x = inp

  if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
  if act == "leaky": x = LeakyReLU(0.2)(x)
  elif act == "prelu": x = PReLU(shared_axes=[1, 2])(x)
  elif act == "relu": x = Activation("relu")(x)
  if dropout: x = Dropout(dropout)(x)

  return x

def conv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:Union[float, None]=None, act:Union[str, None]="leaky", use_bias:bool=True, use_sn:bool=False, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  if use_sn:
    x = ConvSN2D(filters, kernel_size, strides=(strides, strides), padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)
  else:
    x = Conv2D(filters, kernel_size, strides=(strides, strides), padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)

  if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
  if act == "leaky": x = LeakyReLU(0.2)(x)
  elif act == "prelu": x = PReLU(shared_axes=[1, 2])(x)
  elif act == "relu": x = Activation("relu")(x)
  if dropout: x = Dropout(dropout)(x)

  return x

def res_block(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, batch_norm:Union[float, None]=0.5, use_bias:bool=True, use_sn:bool=False, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  gen = inp

  if use_sn:
    model = ConvSN2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)
  else:
    model = Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)
  model = BatchNormalization(momentum=batch_norm, axis=-1)(model)

  model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
  if use_sn:
    model = ConvSN2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(model)
  else:
    model = Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(model)
  model = BatchNormalization(momentum=batch_norm, axis=-1)(model)

  model = Add()(inputs=[gen, model])

  return model

def RRDB(inp, filters:int=64, kernel_size:int=3, batch_norm:Union[float, None]=None, use_bias:bool=True, use_sn:bool=False, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  def dense_block(inp):
    if use_sn:
      x1 = ConvSN2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(inp)
    else:
      x1 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(inp)
    if batch_norm: x1 = BatchNormalization(momentum=batch_norm, axis=-1)(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([inp, x1])

    if use_sn:
      x2 = ConvSN2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x1)
    else:
      x2 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x1)
    if batch_norm: x2 = BatchNormalization(momentum=batch_norm, axis=-1)(x2)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([inp, x1, x2])

    if use_sn:
      x3 = ConvSN2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x2)
    else:
      x3 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x2)
    if batch_norm: x3 = BatchNormalization(momentum=batch_norm, axis=-1)(x3)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([inp, x1, x2, x3])

    if use_sn:
      x4 = ConvSN2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x3)
    else:
      x4 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x3)
    if batch_norm: x4 = BatchNormalization(momentum=batch_norm, axis=-1)(x4)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([inp, x1, x2, x3, x4])

    if use_sn:
      x5 = ConvSN2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x4)
    else:
      x5 = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)(x4)
    if batch_norm: x5 = BatchNormalization(momentum=batch_norm, axis=-1)(x5)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    x = Add()([x5, inp])
    return x

  x = dense_block(inp)
  x = dense_block(x)
  x = dense_block(x)
  x = Lambda(lambda x: x * 0.2)(x)
  out = Add()([x, inp])
  return out