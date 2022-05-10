import torch

LR = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
IMG_CH = 3
NOISE_DIM = 128

EPOCHS = 88
FEATURES_CRIT = FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GRAD_PENALTY = 10

START_STEP_VAL = 234
SAMPLE_PER_STEPS = 200

MODEL_NAME = "celeb_wgan-gp_model"
NUMBER_OF_SAMPLE_IMAGES = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")