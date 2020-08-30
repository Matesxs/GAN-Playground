from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf
import numpy as np
from colorama import Fore

def create_feature_extractor(input_shape:tuple, layers_to_extract:list):
  if not layers_to_extract: return None

  vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
  vgg.trainable = False

  outputs = []
  for ltx in layers_to_extract:
    try:
      if isinstance(ltx, int):
        layer = vgg.layers[ltx]
      elif isinstance(ltx, str):
        layer = vgg.get_layer(ltx)
      else: raise ValueError(Fore.RED + "Invalid value in feature extraction list" + Fore.RESET)
    except:
      raise Exception(Fore.RED + f"Cant find layer {ltx} in VGG19 network" + Fore.RESET)

    print(Fore.MAGENTA + f"Layer {layer.name} added to feature extractor list" + Fore.RESET)
    outputs.append(layer.output)

  model = Model([vgg.input], outputs, name='feature_extractor')
  model.trainable = False
  return model

def preprocess_vgg(x):
  """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
  if isinstance(x, np.ndarray):
    return preprocess_input((x + 1) * 127.5)
  else:
    return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)