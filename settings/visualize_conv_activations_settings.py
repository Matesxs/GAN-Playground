### General visualization settings ###
VISUALIZATION_MODE = "upscale" # upscale, generator, discriminator
OUTPUT_FOLDER_PATH = "generated_images"

### SR visualization settings ###
SR_INPUT_IMAGE_PATH = r"datasets/testing_image1.png"
INPUT_IMAGE_SHAPE_FOR_UPSCALE = (64, 64, 3)

MODEL_NAME_UPSCALE = "mod_srgan_exp_v2"
WEIGHTS_PATH_UPSCALE = r"training_data/srgan/mod_srgan_exp_v2__mod_base_9layers__(64, 64, 3)_to_(256, 256, 3)/weights/313000/generator_mod_srgan_exp_v2.h5"

### Generator visualization settings ###
LATENT_DIM_FOR_GENERATOR = 128
NUMBER_OF_UPSCALES_FOR_UPSCALE = 2
TARGET_IMAGE_SHAPE_FOR_GENERATOR = (64, 64, 3)

MODEL_NAME_DISCRIMINATOR = "mod_base_9layers"
WEIGHTS_PATH_DISCRIMINATOR = r""

### Discriminator visualization settings ###
DISC_INPUT_IMAGE_PATH = r"datasets/testing_image2.png"
INPUT_IMAGE_SHAPE_FOR_DISCRIMINATOR = (64, 64, 3)

MODEL_NAME_GENERATOR = ""
WEIGHTS_PATH_GENERATOR = r""