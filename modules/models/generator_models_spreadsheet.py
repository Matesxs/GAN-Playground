from typing import Union
from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Dense, Reshape, Conv2D, LeakyReLU, Add, Lambda
from tensorflow import Tensor

from modules.models.custom_layers import deconv_layer, conv_layer, RRDB1
from modules.helpers import count_upscaling_start_size

def mod_testing(inp:Union[Tensor, Layer], image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 3)

  m = Dense(64 * st_s[0] * st_s[1], activation=None)(inp)
  m = LeakyReLU(0.2)(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  skip = m

  m = RRDB1(m, filters=64, kernel_size=3, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act=None, use_bias=True, kernel_initializer=kernel_initializer)
  m = Lambda(lambda x: x * 0.2)(m)
  m = Add()([skip, m])

  for _ in range(3):
    m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=2, act="prelu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  m = Conv2D(image_channels, kernel_size=3, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
  return m

def mod_testing2(inp:Union[Tensor, Layer], image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 4)

  m = Dense(128 * st_s[0] * st_s[1], activation=None)(inp)
  m = LeakyReLU(0.2)(m)
  m = Reshape((st_s[0], st_s[1], 128))(m)

  m = deconv_layer(m, 512, kernel_size=4, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 512, kernel_size=4, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 128, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 128, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 64, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 64, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = Conv2D(image_channels, kernel_size=3, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
  return m

def mod_testing3(inp:Union[Tensor, Layer], image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 6)

  m = Dense(2048 * st_s[0] * st_s[1], activation=None)(inp)
  m = LeakyReLU(0.2)(m)
  m = Reshape((st_s[0], st_s[1], 2048))(m)

  m = deconv_layer(m, 512, kernel_size=4, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 512, kernel_size=4, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 128, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 128, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 64, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 64, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 32, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 32, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 16, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 16, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = Conv2D(image_channels, kernel_size=3, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
  return m

def mod_testing4(inp:Union[Tensor, Layer], image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = Reshape((1, 1, inp.shape[1]))(inp)

  m = deconv_layer(m, 512, kernel_size=4, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 512, kernel_size=4, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 128, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 128, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 64, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 64, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 32, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 32, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = deconv_layer(m, 16, kernel_size=3, upscale_multiplier=2, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)
  m = deconv_layer(m, 16, kernel_size=3, upscale_multiplier=1, act="relu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = Conv2D(image_channels, kernel_size=3, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)

  assert m.shape[1:] == image_shape, "Incompatible model with this target resolution"
  return m

def mod_testing5(inp:Union[Tensor, Layer], image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = Reshape((1, 1, inp.shape[1]))(inp)

  m = conv_layer(m, filters=64, kernel_size=4, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  skip1 = m

  m = RRDB1(m, filters=64, kernel_size=3, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act=None, use_bias=True, kernel_initializer=kernel_initializer)
  m = Lambda(lambda x: x * 0.2)(m)
  m = Add()([skip1, m])

  for _ in range(3):
    m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=2, act="prelu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  skip2 = m

  m = RRDB1(m, filters=64, kernel_size=3, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act=None, use_bias=True, kernel_initializer=kernel_initializer)
  m = Lambda(lambda x: x * 0.2)(m)
  m = Add()([skip2, m])

  for _ in range(3):
    m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=2, act="prelu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  m = Conv2D(image_channels, kernel_size=3, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)

  assert m.shape[1:] == image_shape, "Incompatible model with this target resolution"
  return m

def mod_testing6(inp:Union[Tensor, Layer], image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = Reshape((1, 1, inp.shape[1]))(inp)

  m = conv_layer(m, filters=128, kernel_size=4, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)
  m = conv_layer(m, filters=128, kernel_size=4, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  for _ in range(3):
    m = deconv_layer(m, 64, kernel_size=3, upscale_multiplier=2, act="prelu", batch_norm=None, use_subpixel_conv2d=False, use_bias=True, kernel_initializer=kernel_initializer)

  skip1 = m

  m = RRDB1(m, filters=64, kernel_size=3, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act=None, use_bias=True, kernel_initializer=kernel_initializer)
  m = Lambda(lambda x: x * 0.2)(m)
  m = Add()([skip1, m])

  skip2 = m

  m = RRDB1(m, filters=64, kernel_size=3, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act=None, use_bias=True, kernel_initializer=kernel_initializer)
  m = Lambda(lambda x: x * 0.2)(m)
  m = Add()([skip2, m])

  for _ in range(3):
    m = deconv_layer(m, 256, kernel_size=3, upscale_multiplier=2, act="prelu", batch_norm=None, use_subpixel_conv2d=True, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=32, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  m = Conv2D(image_channels, kernel_size=3, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)

  assert m.shape[1:] == image_shape, "Incompatible model with this target resolution"
  return m