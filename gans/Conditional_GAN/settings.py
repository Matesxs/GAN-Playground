import torch

LR = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
IMG_CH = 1
NUM_OF_CLASSES = 10
NOISE_DIM = 128
EMBED_SIZE = 128

EPOCHS = 20
FEATURES_CRIT = FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GRAD_PENALTY = 10

START_STEP_VAL = 0
SAMPLE_PER_STEPS = 100

MODEL_NAME = "mnist_cond-gan_model"
NUMBER_OF_SAMPLE_IMAGES = 32

GEN_MODEL_WEIGHTS_TO_LOAD = None
CRITIC_MODEL_WEIGHTS_TO_LOAD = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")