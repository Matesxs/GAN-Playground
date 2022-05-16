import torch
from math import log2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "celeba_progan_model"

DATASET_PATH = "datasets/celeba_hq"

START_IMAGE_SIZE = 4
IMG_SIZE = 1024
IMG_CH = 3

GEN_MODEL_WEIGHTS_TO_LOAD = None
CRIT_MODEL_WEIGHTS_TO_LOAD = None

SAVE_CHECKPOINTS = False
CHECKPOINT_EVERY = 10

SAMPLE_EVERY = 1
TESTING_SAMPLES = 32

FEATURES = [512, 512, 512, 512, 256, 128, 64, 32, 16]
Z_DIM = 512

LR = 1e-3
LR_DECAY = False
LR_DECAY_COEF = 0.1
LR_DECAY_EVERY = 50
START_ALPHA = 1e-5

BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]
LAMBDA_GP = 10
NUM_OF_STEPES = int(log2(IMG_SIZE / 4)) + 1
PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZES)

NUM_OF_WORKERS = 4
