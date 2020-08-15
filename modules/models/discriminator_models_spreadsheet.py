from keras.layers import Layer, Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Initializer, RandomNormal

from modules.models.custom_layers import conv_layer

def mod_base_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = conv_layer(inp, 32, kernel_size=3, strides=2, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 64, kernel_size=3, strides=2, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 128, kernel_size=3, strides=2, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 256, kernel_size=3, strides=1, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 512, kernel_size=3, strides=1, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = Flatten()(m)
  return m

def mod_ext_5layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = conv_layer(inp, 64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 128, kernel_size=3, strides=2, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 256, kernel_size=3, strides=2, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 512, kernel_size=3, strides=2, dropout=0.25, batch_norm=None, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 1024, kernel_size=3, strides=2, dropout=0.25, batch_norm=None, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = Flatten()(m)
  return m

def mod_base_6layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = conv_layer(inp, 128, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 256, kernel_size=3, strides=1, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 256, kernel_size=3, strides=2, dropout=0.25, batch_norm=0.8, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 512, kernel_size=3, strides=1, dropout=0.25, batch_norm=None, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 512, kernel_size=3, strides=2, dropout=0.25, batch_norm=None, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 1024, kernel_size=3, strides=2, dropout=0.25, batch_norm=None, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = Flatten()(m)
  return m

# Disc for SRGAN
def mod_base_9layers(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = conv_layer(inp, 64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 64, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 128, kernel_size=3, strides=1, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 128, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 256, kernel_size=3, strides=1, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 256, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 512, kernel_size=3, strides=1, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 512, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, kernel_initializer=kernel_initializer)

  m = GlobalAveragePooling2D()(m)
  m = Dropout(0.4)(m)

  m = Dense(1024, activation=None)(m)
  m = LeakyReLU(0.2)(m)
  return m

# Disc for SRGAN
def mod_base_9layers_sn(inp:Layer, kernel_initializer:Initializer=RandomNormal(stddev=0.02)):
  m = conv_layer(inp, 64, kernel_size=3, strides=1, dropout=None, batch_norm=None, act="leaky", use_bias=True, use_sn=True, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 64, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, use_sn=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 128, kernel_size=3, strides=1, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, use_sn=True, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 128, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, use_sn=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 256, kernel_size=3, strides=1, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, use_sn=True, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 256, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, use_sn=True, kernel_initializer=kernel_initializer)

  m = conv_layer(m, 512, kernel_size=3, strides=1, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, use_sn=True, kernel_initializer=kernel_initializer)
  m = conv_layer(m, 512, kernel_size=3, strides=2, dropout=None, batch_norm=0.5, act="leaky", use_bias=False, use_sn=True, kernel_initializer=kernel_initializer)

  m = GlobalAveragePooling2D()(m)
  m = Dropout(0.4)(m)

  m = Dense(1024, activation=None)(m)
  m = LeakyReLU(0.2)(m)
  return m