# Settings of classic gans
DATASET_PATH = "datasets/dogs_normalized__64x64"

NUM_OF_TRAINING_EPOCHS = 200

LATENT_DIM = 128
BATCH_SIZE = 16
BUFFERED_BATCHES = 20

START_EPISODE = 0

GEN_MODEL = "mod_ext2_4upscl"
GEN_WEIGHTS = None # "training_data/dcgan/mod_ext_4upscl__mod_ext_5layers__5pt/weights/350/generator_mod_ext_4upscl.h5"
DISC_MODEL = "mod_ext_5layers"
DICS_WEIGHTS = None # "training_data/dcgan/mod_ext_4upscl__mod_ext_5layers__5pt/weights/350/discriminator_mod_ext_5layers.h5"

# Settings of SR gans
DATASET_SR_PATH = "datasets/faces_normalized__256x256"

NUM_OF_TRAINING_EPOCHS_SR = 20

BATCH_SIZE_SR = 4
CUSTOM_BATCHES_PER_EPOCH = 5_000
BUFFERED_BATCHES_SR = 20

START_EPISODE_SR = 0

NUM_OF_UPSCALES = 2
GEN_SR_MODEL = "mod_srgan_base"
GEN_SR_WEIGHTS = None
DISC_SR_MODEL = "mod_base_9layers"
DICS_SR_WEIGHTS = None

# General settings
LOAD_FROM_CHECKPOINTS = True
CHECK_DATASET = False # Leave this false only when you are sure your dataset is consistent

WEIGHTS_SAVE_INTERVAL = 5
PROGRESS_IMAGE_SAVE_INTERVAL = 1
SAVE_RAW_IMAGES = True # Save progress images to folder too (if false then they will be saved only to tensorboard)

PATH_TO_PYTHON_EXECUTABLE = "venv/Scripts/python.exe" # Only for starting tensorboard for example from venv