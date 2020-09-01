import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdin = stdin
sys.stderr = stderr

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except:
    pass

from keras.applications.vgg19 import VGG19

vgg = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
vgg.trainable = False

for idx, layer in enumerate(vgg.layers):
  if "input" not in layer.name:
    print(f"{idx} - {layer.name}")