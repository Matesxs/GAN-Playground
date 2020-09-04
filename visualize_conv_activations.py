import os
import sys
import shutil
from cv2 import cv2 as cv
import numpy as np

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

from keras import Model
from keras.layers import Input

from modules.models import upscaling_generator_models_spreadsheet, discriminator_models_spreadsheet, generator_models_spreadsheet
from settings.visualize_conv_activations_settings import *

def calculate_grid(num_of_images):
  row_size = num_of_images
  col_size = 1
  
  while True:
    if (row_size / 2) < (col_size * 2): break
    row_size = row_size // 2
    col_size *= 2

  return col_size, row_size

if VISUALIZATION_MODE == "upscale":
  model_input = Input(shape=INPUT_IMAGE_SHAPE_FOR_UPSCALE)

  try:
    m = getattr(upscaling_generator_models_spreadsheet, MODEL_NAME_UPSCALE)(model_input, INPUT_IMAGE_SHAPE_FOR_UPSCALE, NUMBER_OF_UPSCALES_FOR_UPSCALE)
  except Exception as e:
    raise Exception(f"Generator model not found!\n{e}")

  base_model = Model(model_input, m)
  base_model.load_weights(WEIGHTS_PATH_UPSCALE)
elif VISUALIZATION_MODE == "generator":
  model_input = Input(shape=(LATENT_DIM_FOR_GENERATOR,))

  try:
    m = getattr(generator_models_spreadsheet, MODEL_NAME_GENERATOR)(model_input, TARGET_IMAGE_SHAPE_FOR_GENERATOR, TARGET_IMAGE_SHAPE_FOR_GENERATOR)
  except Exception as e:
    raise Exception(f"Generator model not found!\n{e}")

  base_model = Model(model_input, m)
  base_model.load_weights(WEIGHTS_PATH_GENERATOR)
elif VISUALIZATION_MODE == "discriminator":
  model_input = Input(shape=INPUT_IMAGE_SHAPE_FOR_DISCRIMINATOR)

  try:
    m = getattr(discriminator_models_spreadsheet, MODEL_NAME_DISCRIMINATOR)(model_input)
  except Exception as e:
    raise Exception(f"Discriminator model not found!\n{e}")

  base_model = Model(model_input, m)
  base_model.load_weights(WEIGHTS_PATH_DISCRIMINATOR)
else:
  raise Exception("Unknown visualization mode")

output_layers = [layer.output for layer in base_model.layers if "conv" in layer.name.lower()]
output_layer_names = [f"{idx}layer_{layer.name}" for idx, layer in enumerate(base_model.layers) if "conv" in layer.name.lower()]
assert len(output_layers) > 0

vis_model = Model(base_model.input, output_layers)

if os.path.exists(OUTPUT_FOLDER_PATH): shutil.rmtree(OUTPUT_FOLDER_PATH, True)
os.makedirs(OUTPUT_FOLDER_PATH)

def make_and_save_images(activations, output_layer_names):
  for activation, layer_name in zip(activations, output_layer_names):
    activation_shape = activation.shape
    grid_sizes = calculate_grid(activation_shape[3])

    final_image = np.zeros((activation_shape[1] * grid_sizes[0], activation_shape[2] * grid_sizes[1]))
    for col_idx in range(grid_sizes[0]):
      for row_idx in range(grid_sizes[1]):
        final_image[int(col_idx * activation_shape[1]):int((col_idx + 1) * activation_shape[1]), int(row_idx * activation_shape[2]):int((row_idx + 1) * activation_shape[2])] = (0.5 * activation[0, :, :, int((col_idx * grid_sizes[1]) + row_idx)] + 0.5) * 255

    cv.imwrite(os.path.join(OUTPUT_FOLDER_PATH, f"{layer_name}.png"), np.array(final_image))

if VISUALIZATION_MODE == "upscale":
  lr_image = cv.imread(SR_INPUT_IMAGE_PATH)

  if lr_image.shape != INPUT_IMAGE_SHAPE_FOR_UPSCALE:
    lr_image = cv.resize(lr_image, dsize=(INPUT_IMAGE_SHAPE_FOR_UPSCALE[1], INPUT_IMAGE_SHAPE_FOR_UPSCALE[0]), interpolation=(cv.INTER_AREA if (lr_image.shape[0] > INPUT_IMAGE_SHAPE_FOR_UPSCALE[0] and lr_image.shape[1] > INPUT_IMAGE_SHAPE_FOR_UPSCALE[1]) else cv.INTER_CUBIC))

  activations = vis_model.predict(np.array([cv.cvtColor(lr_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0]))
  make_and_save_images(activations[:-1], output_layer_names[:-1])
elif VISUALIZATION_MODE == "generator":
  activations = vis_model.predict(np.random.normal(np.random.normal(0.0, 1.0, size=(1, LATENT_DIM_FOR_GENERATOR))))
  make_and_save_images(activations[:-1], output_layer_names[:-1])
elif VISUALIZATION_MODE == "discriminator":
  image = cv.imread(DISC_INPUT_IMAGE_PATH)

  if image.shape != INPUT_IMAGE_SHAPE_FOR_DISCRIMINATOR:
    image = cv.resize(image, dsize=(INPUT_IMAGE_SHAPE_FOR_DISCRIMINATOR[1], INPUT_IMAGE_SHAPE_FOR_DISCRIMINATOR[0]), interpolation=(cv.INTER_AREA if (image.shape[0] > INPUT_IMAGE_SHAPE_FOR_DISCRIMINATOR[0] and image.shape[1] > INPUT_IMAGE_SHAPE_FOR_DISCRIMINATOR[1]) else cv.INTER_CUBIC))

  activations = vis_model.predict(np.array([cv.cvtColor(image, cv.COLOR_BGR2RGB) / 127.5 - 1.0]))
  make_and_save_images(activations, output_layer_names)