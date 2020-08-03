################################
### Settings of classic GANs ###
################################
# Data source settings
DATASET_PATH = "datasets/dogs_normalized__64x64__train"
TESTING_DATASET_PATH = "datasets/dogs_normalized__64x64__test"

# Training settings
START_EPISODE = 0
NUM_OF_TRAINING_EPISODES = 1_000_000

# Num of episodes after whitch progress image/s will be created to "track" progress of training
PROGRESS_IMAGE_SAVE_INTERVAL = 2_500

BATCH_SIZE = 16
# Num of batches preloaded in buffer
BUFFERED_BATCHES = 100

# Model settings
# Latent dim is size of "tweakable" parameters fed to generator
LATENT_DIM = 128
GEN_MODEL = "mod_ext2_4upscl"
GEN_WEIGHTS = None
DISC_MODEL = "mod_ext_5layers"
DICS_WEIGHTS = None


##########################################
### Settings of SR GANs and SR Resnets ###
##########################################
# Data source settings
DATASET_SR_PATH = "datasets/all_normalized__256x256__train"
TESTING_DATASET_SR_PATH = "datasets/all_normalized__256x256__test"
# If none will be provided then script will select some random one
CUSTOM_HR_TEST_IMAGE = "datasets/testing_image.png"

# Training settings
START_EPISODE_SR = 0
GENERATOR_TRAIN_EPISODES_OF_SRGAN = 1_500_000
# Discriminator need to catch up with generator before it will adding details to image
DISCRIMINATOR_TRAIN_EPISODES_OF_SRGAN = 100_000
COMBINED_TRAINING_EPISODES_SRGAN = 500_000

# Num of episodes after whitch progress image/s will be created to "track" progress of training
PROGRESS_IMAGE_SAVE_INTERVAL_SR = 5_000

# Base LRs
GEN_LR_SRGAN = 1e-4
DISC_LR_SRGAN = 1e-4

# Schedule of LR
GEN_LR_SCHEDULE_SRGAN =  {1_650_000: 1e-5, 1_850_000: 5e-6, 1_900_000: 1e-6, 2_000_000: 1e-7}
DISC_LR_SCHEDULE_SRGAN = {1_650_000: 1e-5, 1_850_000: 5e-6, 1_900_000: 1e-6, 2_000_000: 1e-7}
RESTORE_BEST_PNSR_MODELS_EPISODES = [1_500_000, 1_900_000, 2_000_000]

# Discriminator label noise settings
# Leave as None for not use noise
DISCRIMINATOR_START_NOISE_OF_SRGAN = 0.20
DISCRIMINATOR_NOISE_DECAY_OF_SRGAN = 0.999993
# Noise target where stop decaying
DISCRIMINATOR_TARGET_NOISE_OF_SRGAN = 0

# Discriminator training settings
AUTOBALANCE_TRAINING_OF_SRGAN = True
DISCRIMINATOR_TRAINING_MULTIPLIER = 2

BATCH_SIZE_SR = 4
TESTING_BATCH_SIZE_SR = 32
# Num of batches preloaded in buffer
BUFFERED_BATCHES_SR = 100

# Model settings
# Number of doubling resolution
NUM_OF_UPSCALES = 2
GEN_SR_MODEL = "mod_srgan_exp_sn"
GEN_SR_WEIGHTS = None
DISC_SR_MODEL = "mod_base_9layers_sn"
DICS_SR_WEIGHTS = None

# Saving settings
SAVE_ONLY_BEST_PNSR_WEIGHTS = True


########################
### General settings ###
########################
# Check if you want to load last autocheckpoint (If weights were provided thne checkpoint will be overriden by them)
LOAD_FROM_CHECKPOINTS = True
# Leave this false only when you are sure your dataset is consistent (Check whole dataset if all images have same dimensions before training)
CHECK_DATASET = False

# Num of episodes after whitch weights will be saved (Its not the same as checkpoint!)
WEIGHTS_SAVE_INTERVAL = 10_000
# Save progress images to folder too (if false then they will be saved only to tensorboard)
SAVE_RAW_IMAGES = True
# Duration of one frame if gif is created from progress images after training
GIF_FRAME_DURATION = 100

# Num of worker used to preload data for training/testing
NUM_OF_LOADING_WORKERS = 8

## NOTE ##
# Tensorboard console command: tensorboard --logdir training_data --samples_per_plugin=images=200