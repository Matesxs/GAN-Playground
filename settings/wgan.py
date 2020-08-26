# Data source settings
DATASET_PATH = "datasets/faces_normalized__64x64"

# Training settings
START_EPISODE = 0
NUM_OF_TRAINING_EPISODES = 300_000

# Num of episodes after whitch progress image/s will be created to "track" progress of training
PROGRESS_IMAGE_SAVE_INTERVAL = 500
# Num of episodes after whitch weights will be saved (Its not the same as checkpoint!)
WEIGHTS_SAVE_INTERVAL = 2_500

BATCH_SIZE = 32
# Num of batches preloaded in buffer
BUFFERED_BATCHES = 100

# Model settings
# Latent dim is size of "tweakable" parameters fed to generator
LATENT_DIM = 128
GEN_MODEL = "mod_testing"
GEN_WEIGHTS = None
DISC_MODEL = "mod_testing8"
DICS_WEIGHTS = None

# Check if you want to load last autocheckpoint (If weights were provided thne checkpoint will be overriden by them)
LOAD_FROM_CHECKPOINTS = True
# Leave this false only when you are sure your dataset is consistent (Check whole dataset if all images have same dimensions before training)
CHECK_DATASET = False

# Save progress images to folder too (if false then they will be saved only to tensorboard)
SAVE_RAW_IMAGES = True
# Duration of one frame if gif is created from progress images after training
GIF_FRAME_DURATION = 100

# Num of worker used to preload data for training/testing
NUM_OF_LOADING_WORKERS = 8