from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, Dense, Reshape, Conv2D, Activation, BatchNormalization, Add, PReLU

from modules.models.custom_layers import deconv_layer, res_block

# Calculate start image size based on final image size and number of upscales
def count_upscaling_start_size(target_image_shape:tuple, num_of_upscales:int):
  upsc = (target_image_shape[0] // (2 ** num_of_upscales), target_image_shape[1] // (2 ** num_of_upscales))
  if upsc[0] < 1 or upsc[1] < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
  return upsc

def mod_base_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 3)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 512) -> (4*st_s, 4*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 256) -> (8*st_s, 8*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=2, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, 64)
  m = deconv_layer(m, 64, kernel_size=3, strides=1, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 64) -> (8*st_s, 8*st_s, 32)
  m = deconv_layer(m, 32, kernel_size=3, strides=1, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 32) -> (8*st_s, 8*st_s, num_ch)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_ext_3upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 3)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
  m = deconv_layer(m, 1024, kernel_size=3, strides=2, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=1, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=1, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (8*st_s, 8*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=1, use_subpixel_conv2d=False, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 128) -> (8*st_s, 8*st_s, image_channels)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_base_4upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 4)

  # (64 * st_s^2,) -> (st_s, st_s, 64)
  m = Dense(64 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 64))(m)

  # (st_s, st_s, 64) -> (2*st_s, 2*st_s, 1024)
  m = deconv_layer(m, 1024, kernel_size=3, strides=2, use_subpixel_conv2d=True, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (2*st_s, 2*st_s, 1024) -> (4*st_s, 4*st_s, 512)
  m = deconv_layer(m, 512, kernel_size=3, strides=2, use_subpixel_conv2d=True, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (4*st_s, 4*st_s, 512) -> (8*st_s, 8*st_s, 256)
  m = deconv_layer(m, 256, kernel_size=3, strides=2, use_subpixel_conv2d=True, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (8*st_s, 8*st_s, 256) -> (16*st_s, 16*st_s, 128)
  m = deconv_layer(m, 128, kernel_size=3, strides=2, use_subpixel_conv2d=True, act="relu", batch_norm=0.8, use_bias=False, use_sn=False, kernel_initializer=kernel_initializer)

  # (16*st_s, 16*st_s, 128) -> (16*st_s, 16*st_s, image_channels)
  m = Conv2D(image_channels, kernel_size=(3, 3), padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_exp_4upscl(inp:Layer, image_shape:tuple, image_channels:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  st_s = count_upscaling_start_size(image_shape, 4)

  m = Dense(32 * st_s[0] * st_s[1], kernel_initializer=kernel_initializer, activation=None)(inp)
  m = Activation("relu")(m)
  m = Reshape((st_s[0], st_s[1], 32))(m)

  m = Conv2D(filters=64, kernel_size=9, strides=1, padding="same", kernel_initializer=kernel_initializer, activation=None)(m)
  m = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(m)

  skip = m

  for _ in range(16):
    m = res_block(m, 64, 3, 1, batch_norm=0.5, use_bias=False, kernel_initializer=kernel_initializer)

  m = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_initializer=kernel_initializer, use_bias=False, activation=None)(m)
  m = BatchNormalization(momentum=0.5, axis=-1)(m)
  m = Add()(inputs=[skip, m])

  for _ in range(4):
    m = deconv_layer(m, 256, kernel_size=3, strides=2, act="leaky", batch_norm=None, use_subpixel_conv2d=True, upsample_first=False, use_bias=False, kernel_initializer=kernel_initializer)

  m = Conv2D(filters=image_channels, kernel_size=9, strides=1, padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m