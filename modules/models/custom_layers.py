from typing import Union
import tensorflow as tf
from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Conv2D, UpSampling2D, BatchNormalization, Dropout, Add, PReLU, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation

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
  return Lambda(subpixel, output_shape=subpixel_shape, name=f"cubpixel_conv2d_{subpixel_index}")

def deconv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:Union[float, None]=None, use_subpixel_conv2d:bool=False, leaky:bool=True, upsample_first:bool=True, use_bias:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  if upsample_first:
    if strides > 1:
      if use_subpixel_conv2d:
        x = SubpixelConv2D(2)(inp)
      else:
        x = UpSampling2D(size=strides)(inp)
    else: x = inp

    x = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(x)
  else:
    x = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)

    if strides > 1:
      if use_subpixel_conv2d:
        x = SubpixelConv2D(2)(x)
      else:
        x = UpSampling2D(size=strides)(x)
    else: x = inp

  if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
  if leaky: x = LeakyReLU(0.2)(x)
  else: x = Activation("relu")(x)
  if dropout: x = Dropout(dropout)(x)

  return x

def conv_layer(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, dropout:float=None, batch_norm:Union[float, None]=None, leaky:bool=True, use_bias:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  x = Conv2D(filters, (kernel_size, kernel_size), strides=(strides, strides), padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)

  if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
  if leaky: x = LeakyReLU(0.2)(x)
  else: x = Activation("relu")(x)
  if dropout: x = Dropout(dropout)(x)

  return x

def res_block(inp:Layer, filters:int, kernel_size:int=3, strides:int=2, batch_norm:float=0.5, use_bias:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert filters > 0, "Invalid filter number"
  assert kernel_size > 0, "Invalid kernel size"
  assert strides > 0, "Invalid stride size"

  gen = inp

  model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)
  model = BatchNormalization(momentum=batch_norm, axis=-1)(model)

  model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
  model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(model)
  model = BatchNormalization(momentum=batch_norm, axis=-1)(model)

  model = Add()(inputs=[gen, model])

  return model

def identity_layer(inp, filters_number_list:Union[list, int], kernel_size:int=3, dropout:float=None, batch_norm:Union[float, None]=None, use_bias:bool=True, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  assert kernel_size > 0, "Invalid kernel size"

  x = inp
  if isinstance(filters_number_list, list):
    for index, filters in enumerate(filters_number_list):
      assert filters > 0, "Invalid filter number"
      if index > 0:
        x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(x)

      x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(x)

      if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
      if dropout: x = Dropout(dropout)(x)
  else:
    assert filters_number_list > 0, "Invalid filter number"

    x = Conv2D(filters_number_list, kernel_size=(kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(x)

    if batch_norm: x = BatchNormalization(momentum=batch_norm, axis=-1)(x)
    if dropout: x = Dropout(dropout)(x)

  if inp.shape != x.shape:
    inp = Conv2D(x.shape[-1], kernel_size=1, strides=1, padding="valid", kernel_initializer=kernel_initializer, use_bias=use_bias, activation=None)(inp)

  x = Add()(inputs=[x, inp])
  return x