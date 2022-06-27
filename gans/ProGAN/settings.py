import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "celeba_progan_model"

DATASET_PATH = "datasets/celeb_normalized__256x256"

TARGET_IMAGE_SIZE = 256
IMG_CH = 3

OVERRIDE_ITERATION = None
GEN_MODEL_WEIGHTS_TO_LOAD = None
CRIT_MODEL_WEIGHTS_TO_LOAD = None

SAVE_CHECKPOINTS = True
CHECKPOINT_EVERY = 5_000

STATS_SAMPLE_EVERY = 100
IMAGE_SAMPLE_EVERY = 2_000
IMAGE_TESTING_SAMPLES = 32

# Originals
# BASE_FEATURES = 8192
# FEATURES_MAX = 512
# Z_DIM = 512
BASE_FEATURES = 4096
FEATURES_MAX = 256
Z_DIM = 256

LR = 1e-4 # 1e-3
START_ALPHA = 1e-5
IMG_SIZE_TO_BATCH_SIZE = { 4:32, 8:32, 16:16, 32:16, 64:8, 128:4, 256:4 }
LAMBDA_GP = 10 # 10
EPSILON_DRIFT = 0.001 # 0.001

BATCH_REPEATS = 4
CRITIC_ACCUMULATION_STEPS = 1
GENERATOR_ACCUMULATION_STEPS = 1

# Fading steps, full training steps
#                               4               8                16                32                 64               128                 256
PROGRESSIVE_ITERATIONS = [(0, 60_000), (30_000, 30_000), (50_000, 50_000), (50_000, 50_000), (100_000, 100_000), (200_000, 200_000), (400_000, 400_000)]
ADDITIONAL_TRAINING = 200_000
NUMBER_OF_STEPS = int(math.log2(TARGET_IMAGE_SIZE)) - 1
assert NUMBER_OF_STEPS >= len(PROGRESSIVE_ITERATIONS), "Specified iterations for layers that are not defined in model"

NUM_OF_WORKERS = 8

INCEPTION_SCORE_BATCH_SIZE = 32
INCEPTION_SCORE_NUMBER_OF_BATCHES = 40
INCEPTION_SCORE_SPLIT = 32
