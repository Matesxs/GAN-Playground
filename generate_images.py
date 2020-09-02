import os
import sys
import time
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdin = stdin
sys.stderr = stderr

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except:
    pass

from keras.models import Model
from keras.layers import Input
import numpy as np
from cv2 import cv2 as cv

from modules.models import generator_models_spreadsheet
from settings.generate_images_settings import *

lat_input = Input(shape=(LATENT_DIM,))
preq_gen = getattr(generator_models_spreadsheet, GENERATOR_MODEL_NAME)(lat_input, (TARGET_SHAPE[1], TARGET_SHAPE[0]), TARGET_SHAPE[2])
gen_mod = Model(lat_input, preq_gen, name="generator")

if GENERATOR_WEIGHTS_PATH: gen_mod.load_weights(GENERATOR_WEIGHTS_PATH)

def generate_images():
  gen_images = []

  while len(gen_images) != NUM_OF_GENERATED_IMAGES:
    missing_images = NUM_OF_GENERATED_IMAGES - len(gen_images)
    noise = np.random.normal(np.random.normal(0.0, 1.0, size=(missing_images, LATENT_DIM)))
    images = gen_mod.predict(noise)

    for image in images:
      gen_images.append(image)

  return (np.array(gen_images) + 1) * 127.5

gen_images = generate_images()

if os.path.exists(OUTPUT_FOLDER_PATH): shutil.rmtree(OUTPUT_FOLDER_PATH, True)
os.makedirs(OUTPUT_FOLDER_PATH)

for image in gen_images:
  cv.imwrite(f"{OUTPUT_FOLDER_PATH}/{str(time.time()).replace('.', '_')}.png", cv.cvtColor(image, cv.COLOR_BGR2RGB))