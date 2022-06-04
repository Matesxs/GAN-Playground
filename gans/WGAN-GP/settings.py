import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LR = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
IMG_CH = 1
NOISE_DIM = 128

ITERATIONS = 100_000
FEATURES_CRIT = 128
FEATURES_GEN = 128
CRITIC_ITERATIONS = 5
LAMBDA_GRAD_PENALTY = 10

SAVE_CHECKPOINT = True
CHECKPOINT_EVERY = 5_000

SAMPLE_EVERY = 1_000

DATASET_PATH = "datasets/SOCOFing/Real"

MODEL_NAME = "socofing_wgan-gp_model_pix_shuf2"
NUMBER_OF_SAMPLE_IMAGES = 32

GEN_MODEL_WEIGHTS_TO_LOAD = None
CRITIC_MODEL_WEIGHTS_TO_LOAD = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_OF_WORKERS = 8

transform = A.Compose(
  [
    A.Resize(width=IMG_SIZE, height=IMG_SIZE, interpolation=Image.BICUBIC),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5 for _ in range(IMG_CH)], std=[0.5 for _ in range(IMG_CH)]),
    ToTensorV2()
  ]
)
