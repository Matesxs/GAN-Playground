import torch

LR = 2e-4
BATCH_SIZE = 128
IMG_SIZE = 128
IMG_CH = 3
NOISE_DIM = 128

EPOCHS = 100
FEATURES_DISC = FEATURES_GEN = 64

START_STEP_VAL = 0

MODEL_NAME = "celeb_BIG_dcgan_model"
NUMBER_OF_SAMPLE_IMAGES = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")