DATASET_PATH = "datasets/dogs_normalized__64x64"
LATENT_DIM = 128
BATCH_SIZE = 16
BUFFERED_BATCHES = 20

START_EPISODE = 0

GEN_MODEL = "mod_ext2_4upscl"
GEN_WEIGHTS = None # "training_data/dcgan/mod_ext_4upscl__mod_ext_5layers__5pt/weights/350/generator_mod_ext_4upscl.h5"
DISC_MODEL = "mod_ext_5layers"
DICS_WEIGHTS = None # "training_data/dcgan/mod_ext_4upscl__mod_ext_5layers__5pt/weights/350/discriminator_mod_ext_5layers.h5"
LOAD_FROM_CHECKPOINTS = True

NUM_OF_TRAINING_EPOCHS = 200

WEIGHTS_SAVE_INTERVAL = 5
PROGRESS_IMAGE_SAVE_INTERVAL = 1