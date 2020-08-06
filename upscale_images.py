from keras.models import Model
from keras.layers import Input
import cv2 as cv
import numpy as np

from modules.models import upscaling_generator_models_spreadsheet

START_IMAGE_SHAPE = (64, 64, 3)
NUM_OF_UPSCALES = 2
MODEL_ARCHITECTURE = "mod_srgan_base_sub"
MODEL_WEIGHTS_PATH = r"F:\Projekty\Python\GANTest\training_data\srgan_v2\mod_srgan_base_sub__mod_base_9layers__(64, 64, 3)_to_(256, 256, 3)\weights\140\generator_mod_srgan_base_sub.h5"
INPUT_IMAGE_PATH = r"F:\Projekty\Python\GANTest\datasets\faces_normalized__256x256\110.png"

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
if original_image.shape != START_IMAGE_SHAPE: original_image = cv.resize(original_image, dsize=(START_IMAGE_SHAPE[0], START_IMAGE_SHAPE[1]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > START_IMAGE_SHAPE[0] and original_image.shape[1] > START_IMAGE_SHAPE[1]) else cv.INTER_CUBIC))
input_image = np.array([cv.cvtColor(original_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0])
upscaled_image = model.predict(input_image)[0]
upscaled_image = (0.5 * upscaled_image + 0.5) * 255
upscaled_image = cv.cvtColor(upscaled_image, cv.COLOR_RGB2BGR)
cv.imwrite("upscaled_image.png", upscaled_image)