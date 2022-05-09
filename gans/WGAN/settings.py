import torch

LR = 5e-5
BATCH_SIZE = 64
IMG_SIZE = 64
IMG_CH = 3
NOISE_DIM = 128

EPOCHS = 70
FEATURES_CRIT = FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

START_STEP_VAL = 638
SAMPLE_PER_STEPS = 200

MODEL_NAME = "celeb_wgan_model"
NUMBER_OF_SAMPLE_IMAGES = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")