from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Dense, BatchNormalization, Reshape, Conv2D, Activation

from modules.models.custom_layers import deconv_layer, identity_layer

# Calculate start image size based on final image size and number of upscales
def count_upscaling_start_size(target_image_shape: tuple, num_of_upscales: int):
  upsc = (target_image_shape[0] // (2 ** num_of_upscales), target_image_shape[1] // (2 ** num_of_upscales))
  if upsc[0] < 1 or upsc[1] < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
  return upsc

def mod_base_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 3)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)
  m = BatchNormalization(momentum=0.8)(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 512) -> (4*st_s, 4*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 256) -> (8*st_s, 8*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, 64)
  m = deconv_layer(m, 64, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 64) -> (8*st_s, 8*st_s, 32)
  m = deconv_layer(m, 32, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 32) -> (8*st_s, 8*st_s, num_ch)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_ext_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 3)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)
  m = BatchNormalization(momentum=0.8)(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
  m = deconv_layer(m, 1024, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, image_channels)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_base_4upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 4)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)
  m = BatchNormalization(momentum=0.8)(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
  m = deconv_layer(m, 1024, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (16*st_s, 16*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (16*st_s, 16*st_s, 128) -> (16*st_s, 16*st_s, image_channels)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_ext_4upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 4)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)
  m = BatchNormalization(momentum=0.8)(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
  m = deconv_layer(m, 1024, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (16*st_s, 16*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (16*st_s, 16*st_s, 128) -> (16*st_s, 16*st_s, 128)
  m = identity_layer(m, [64, 128], kernel_size=3, dropout=None, batch_norm=0.8, leaky=False, kernel_initializer=kernel_initializer)

  # (16*st_s, 16*st_s, 128) -> (16*st_s, 16*st_s, image_channels)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_ext2_4upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 4)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)
  m = BatchNormalization(momentum=0.8)(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
  m = deconv_layer(m, 1024, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=1, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (16*st_s, 16*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=2, conv_transpose=False, leaky=False, batch_norm=0.8, kernel_initializer=kernel_initializer)

  # (16*st_s, 16*st_s, 128) -> (16*st_s, 16*st_s, 128)
  m = identity_layer(m, [128, 128], kernel_size=3, dropout=None, batch_norm=0.8, leaky=False, kernel_initializer=kernel_initializer)

  # (16*st_s, 16*st_s, 128) -> (16*st_s, 16*st_s, 128)
  m = identity_layer(m, [64, 128], kernel_size=3, dropout=None, batch_norm=0.8, leaky=False, kernel_initializer=kernel_initializer)

  # (16*st_s, 16*st_s, 128) -> (16*st_s, 16*st_s, image_channels)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m