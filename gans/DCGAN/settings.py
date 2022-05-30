import torch

LR = 2e-4
BATCH_SIZE = 128
IMG_SIZE = 128
IMG_CH = 3
NOISE_DIM = 128

ITERATIONS = 100_000
FEATURES_DISC = FEATURES_GEN = 64

SAVE_CHECKPOINT = False
CHECKPOINT_EVERY = 1_000

SAMPLE_INTERVAL = 200

DATASET_PATH = "datasets/celeb_normalized__128x128"

MODEL_NAME = "celeb_BIG_dcgan_model"
NUMBER_OF_SAMPLE_IMAGES = 16

GEN_MODEL_WEIGHTS_TO_LOAD = None
DISC_MODEL_WEIGHTS_TO_LOAD = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")