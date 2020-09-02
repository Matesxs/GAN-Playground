from keras.models import Model
from keras.layers import Input
from cv2 import cv2 as cv
import numpy as np
import os
import shutil
import time

from modules.models import upscaling_generator_models_spreadsheet
from settings.upscale_images_settings import *

def build_generator():
  small_image_input = Input(shape=START_IMAGE_SHAPE)

  try:
    m = getattr(upscaling_generator_models_spreadsheet, MODEL_ARCHITECTURE)(small_image_input, START_IMAGE_SHAPE, NUM_OF_UPSCALES)
  except Exception as e:
    raise Exception(f"Generator model not found!\n{e}")

  return Model(small_image_input, m, name="generator_model")

model = build_generator()
model.load_weights(MODEL_WEIGHTS_PATH)

original_image = cv.imread(INPUT_IMAGE_PATH)
if original_image.shape != START_IMAGE_SHAPE:
  original_image = cv.resize(original_image, dsize=(START_IMAGE_SHAPE[1], START_IMAGE_SHAPE[0]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > START_IMAGE_SHAPE[0] and original_image.shape[1] > START_IMAGE_SHAPE[1]) else cv.INTER_CUBIC))

input_image = np.array([cv.cvtColor(original_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0])

upscaled_image = model.predict(input_image)[0]
upscaled_image = (0.5 * upscaled_image + 0.5) * 255

if os.path.exists(OUTPUT_FOLDER_PATH): shutil.rmtree(OUTPUT_FOLDER_PATH, True)
os.makedirs(OUTPUT_FOLDER_PATH)

cv.imwrite(f"{os.path.join(OUTPUT_FOLDER_PATH, str(time.time()).replace('.', '_'))}.png", cv.cvtColor(upscaled_image, cv.COLOR_RGB2BGR))