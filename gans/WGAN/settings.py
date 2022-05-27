import torch

LR = 5e-5
BATCH_SIZE = 64
IMG_SIZE = 64
IMG_CH = 3
NOISE_DIM = 128

EPOCHS = 100
FEATURES_CRIT = FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

SAMPLE_EVERY = 200

DATASET_PATH = "datasets/celeb_normalized__64x64"

MODEL_NAME = "celeb_wgan_model"
NUMBER_OF_SAMPLE_IMAGES = 32

GEN_MODEL_WEIGHTS_TO_LOAD = None
CRITIC_MODEL_WEIGHTS_TO_LOAD = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")