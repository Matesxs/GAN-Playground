import torch

LR = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
IMG_CH = 3
NOISE_DIM = 128

EPOCHS = 100
FEATURES_CRIT = FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GRAD_PENALTY = 10

SAMPLE_EVERY = 1

DATASET_PATH = "datasets/celeb_normalized__64x64"

MODEL_NAME = "celeb_wgan-gp_model"
NUMBER_OF_SAMPLE_IMAGES = 32

GEN_MODEL_WEIGHTS_TO_LOAD = None
CRITIC_MODEL_WEIGHTS_TO_LOAD = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")