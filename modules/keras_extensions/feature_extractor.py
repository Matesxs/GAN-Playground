from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf
import numpy as np

def create_feature_extractor(input_shape:tuple, layers_to_extract:list):
  assert len(layers_to_extract) > 0, "Specify layers to extract features"

  vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
  vgg.trainable = False

  outputs = [vgg.layers[i].output for i in layers_to_extract]
  model = Model([vgg.input], outputs, name='feature_extractor')
  model.trainable = False
  return model

def preprocess_vgg(x):
  """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
  if isinstance(x, np.ndarray):
    return preprocess_input((x + 1) * 127.5)
  else:
    return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)