# Settings of classic GANs
DATASET_PATH = "datasets/dogs_normalized__64x64"

NUM_OF_TRAINING_EPOCHS = 10_000

LATENT_DIM = 128
BATCH_SIZE = 16
BUFFERED_BATCHES = 50

START_EPISODE = 0

GEN_MODEL = "mod_ext2_4upscl"
GEN_WEIGHTS = None # "training_data/dcgan/mod_ext_4upscl__mod_ext_5layers__5pt/weights/350/generator_mod_ext_4upscl.h5"
DISC_MODEL = "mod_ext_5layers"
DICS_WEIGHTS = None # "training_data/dcgan/mod_ext_4upscl__mod_ext_5layers__5pt/weights/350/discriminator_mod_ext_5layers.h5"

# Settings of SR GANs and SR Resnets
DATASET_SR_PATH = "datasets/all_normalized__256x256" # "datasets/faces_normalized__256x256"
CUSTOM_HR_TEST_IMAGE = r"F:\Projekty\Python\GANTest\datasets\all_normalized__256x256\60165.png"

NUM_OF_TRAINING_EPOCHS_SR = 10_000
PRETRAIN_EPISODES_OF_SRGAN = 200

BATCH_SIZE_SR = 4
CUSTOM_BATCHES_PER_EPOCH = 5_000
BUFFERED_BATCHES_SR = 50

START_EPISODE_SR = 0

NUM_OF_UPSCALES = 2
GEN_SR_MODEL = "mod_srgan_base_sub" # mod_srgan_base, mod_srgan_ext, mod_srgan_base_sub
GEN_SR_WEIGHTS = None
DISC_SR_MODEL = "mod_base_9layers"
DICS_SR_WEIGHTS = None

# General settings
NUM_OF_TEST_BATCHES = 5

LOAD_FROM_CHECKPOINTS = True
CHECK_DATASET = False # Leave this false only when you are sure your dataset is consistent

WEIGHTS_SAVE_INTERVAL = 5
PROGRESS_IMAGE_SAVE_INTERVAL = 1
SAVE_RAW_IMAGES = True # Save progress images to folder too (if false then they will be saved only to tensorboard)
GIF_FRAME_DURATION = 100