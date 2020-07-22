from keras.initializers import Initializer, RandomNormal
from keras.layers import Layer, BatchNormalization, Conv2D, PReLU, Add, Lambda

from modules.models.custom_layers import deconv_layer, res_block, RRDB, conv_layer

def mod_srgan_base(inp:Layer, start_image_shape:tuple, num_of_upscales:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = Conv2D(filters=64, kernel_size=9, strides=1, padding="same", kernel_initializer=kernel_initializer, input_shape=start_image_shape, activation=None)(inp)
  m = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(m)

  skip = m

  for _ in range(16):
    m = res_block(m, 64, 3, 1, batch_norm=0.5, use_bias=False, kernel_initializer=kernel_initializer)

  m = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_initializer=kernel_initializer, use_bias=False, activation=None)(m)
  m = BatchNormalization(momentum=0.5, axis=-1)(m)
  m = Add()(inputs=[skip, m])

  for _ in range(num_of_upscales):
    m = deconv_layer(m, 256, kernel_size=3, strides=2, act="leaky", batch_norm=None, use_subpixel_conv2d=True, upsample_first=False, use_bias=False, kernel_initializer=kernel_initializer)

  m = Conv2D(filters=start_image_shape[2], kernel_size=9, strides=1, padding="same", activation="tanh", kernel_initializer=kernel_initializer, use_bias=False)(m)
  return m

def mod_srgan_exp(inp:Layer, start_image_shape:tuple, num_of_upscales:int, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = conv_layer(inp, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  skip = m

  m = RRDB(m, 64, kernel_size=3, batch_norm=None, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act=None, use_bias=True, kernel_initializer=kernel_initializer)
  m = Lambda(lambda x: x * 0.2)(m)
  m = Add()([skip, m])

  for _ in range(num_of_upscales):
    m = deconv_layer(m, 256, kernel_size=3, strides=2, act="prelu", batch_norm=None, use_subpixel_conv2d=True, upsample_first=False, use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, filters=64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  m = Conv2D(filters=start_image_shape[2], kernel_size=3, strides=1, padding="same", activation="tanh", kernel_initializer=kernel_initializer)(m)
  return m