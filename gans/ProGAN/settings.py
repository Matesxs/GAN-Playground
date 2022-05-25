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

SAVE_CHECKPOINTS = True
CHECKPOINT_EVERY = 10

SAMPLE_EVERY = 1
TESTING_SAMPLES = 32

FEATURES = [512, 512, 512, 512, 256, 128, 64, 32, 16] # For generator, its reversed in critic
Z_DIM = 512

STEP_OVERRIDE = None

LR = 1e-3
LR_DECAY = False
LR_DECAY_COEF = 0.1
LR_DECAY_EVERY = 50
START_ALPHA = 1e-5
IMG_SIZE_TO_BATCH_SIZE = { 4:64, 16:32, 32:32, 64:16, 128:8, 256:4, 512:2, 1024:2 }
LAMBDA_GP = 10
NUM_OF_STEPES = int(log2(IMG_SIZE / 4)) + 1
PROGRESSIVE_EPOCHS = [200, 150, 100, 75, 50, 50, 50, 50]

NUM_OF_WORKERS = 8
