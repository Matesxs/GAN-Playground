################################
### Settings of classic GANs ###
################################
DATASET_PATH = "datasets/dogs_normalized__64x64__train"
TESTING_DATASET_PATH = "datasets/dogs_normalized__64x64__test"

NUM_OF_TRAINING_EPISODES = 1_000_000

LATENT_DIM = 128
BATCH_SIZE = 16
BUFFERED_BATCHES = 50

START_EPISODE = 0

GEN_MODEL = "mod_ext2_4upscl"
GEN_WEIGHTS = None
DISC_MODEL = "mod_ext_5layers"
DICS_WEIGHTS = None


##########################################
### Settings of SR GANs and SR Resnets ###
##########################################
DATASET_SR_PATH = "datasets/all_normalized__256x256"
TESTING_DATASET_SR_PATH = None
# If none will be provided then script will select some random one
CUSTOM_HR_TEST_IMAGE = r"F:\Projekty\Python\GANTest\datasets\all_normalized__256x256\60165.png"

# Discriminator will be trained more on images where its more behind
AUTOBALANCE_TRAINING_OF_SRGAN = True

COMBINED_TRAINING_EPISODES_SRGAN = 100_000
GENERATOR_TRAIN_EPISODES_OF_SRGAN = 1_500_000
# With generator too (Discriminator need to catch up with generator before it will adding details to image)
DISCRIMINATOR_TRAIN_EPISODES_OF_SRGAN = 500_000
FINETUNE_TRAIN_EPISODES_OF_SRGAN = 200_000

BATCH_SIZE_SR = 4
BUFFERED_BATCHES_SR = 50

START_EPISODE_SR = 0

NUM_OF_UPSCALES = 2
GEN_SR_MODEL = "mod_srgan_base_sub"
GEN_SR_WEIGHTS = None
DISC_SR_MODEL = "mod_base_9layers"
DICS_SR_WEIGHTS = None


########################
### General settings ###
########################
NUM_OF_TEST_BATCHES = 10

LOAD_FROM_CHECKPOINTS = True
# Leave this false only when you are sure your dataset is consistent
CHECK_DATASET = False

WEIGHTS_SAVE_INTERVAL = 10_000
PROGRESS_IMAGE_SAVE_INTERVAL = 2_500
# Save progress images to folder too (if false then they will be saved only to tensorboard)
SAVE_RAW_IMAGES = True
GIF_FRAME_DURATION = 100

NUM_OF_LOADING_WORKERS = 8