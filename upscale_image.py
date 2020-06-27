from keras.models import Model
from keras.layers import Input
import cv2 as cv
import numpy as np

from modules.models import upscaling_generator_models_spreadsheet

START_IMAGE_SHAPE = (64, 64, 3)
NUM_OF_UPSCALES = 2
MODEL_ARCHITECTURE = "mod_srgan_base_sub"
MODEL_WEIGHTS_PATH = r"F:\Projekty\Python\GANTest\training_data\srgan_v2\mod_srgan_base_sub__mod_base_9layers__(64, 64, 3)_to_(256, 256, 3)\weights\15\generator_mod_srgan_base_sub.h5"
INPUT_IMAGE_PATH = r"F:\Projekty\Python\GANTest\datasets\testing_images_normalized__64x64\1.png"

def build_generator():
  small_image_input = Input(shape=START_IMAGE_SHAPE)

  try:
    m = getattr(upscaling_generator_models_spreadsheet, MODEL_ARCHITECTURE)(small_image_input, START_IMAGE_SHAPE,
                                                                    NUM_OF_UPSCALES)
  except Exception as e:
    raise Exception(f"Generator model not found!\n{e}")

  return Model(small_image_input, m, name="generator_model")

model = build_generator()
model.load_weights(MODEL_WEIGHTS_PATH)
input_image = np.array([cv.cvtColor(cv.imread(INPUT_IMAGE_PATH), cv.COLOR_BGR2RGB) / 127.5 - 1.0])
upscaled_image = model.predict(input_image)[0]
upscaled_image = (0.5 * upscaled_image + 0.5) * 255
upscaled_image = cv.cvtColor(upscaled_image, cv.COLOR_RGB2BGR)
cv.imwrite("upscaled_image.png", upscaled_image)