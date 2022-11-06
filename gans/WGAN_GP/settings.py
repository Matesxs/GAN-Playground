import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LR = 1e-4
DECAY_LR = True
DECAY_AFTER_ITERATIONS = 100_000
DECAY_ITERATION = 100_000
BATCH_SIZE = 32
ITERATIONS = 200_000
CRITIC_ITERATIONS = 5
GENERATOR_ITERATIONS = 2
LAMBDA_GRAD_PENALTY = 10

IMG_SIZE = 128
IMG_CH = 3
NOISE_DIM = 512

FEATURES_CRIT = [32, 64, 128, 256, 512]
STRIDES_CRIT = [2, 2, 2, 2, 2]
GENERATOR_USE_PIXELSHUFFLE = True
FEATURES_GEN = [1024, 512, 256, 128, 64, 32]
SCALE_OR_STRIDE_FACTOR_GEN = [2, 2, 2, 2, 2, 2]

SAVE_CHECKPOINT = True
CHECKPOINT_EVERY = 2_000

SAMPLE_EVERY = 100

DATASET_PATH = "C:/HighDemandProjects/datasets/day"

MODEL_NAME = "day_128s_32b_1e4l_5ci_1gi__standard_model_pixs"
NUMBER_OF_SAMPLE_IMAGES = 32

GEN_MODEL_WEIGHTS_TO_LOAD = None
CRITIC_MODEL_WEIGHTS_TO_LOAD = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_OF_WORKERS = 4

transform = A.Compose(
  [
    A.SmallestMaxSize(max_size=IMG_SIZE, interpolation=Image.BICUBIC),
    A.CenterCrop(IMG_SIZE, IMG_SIZE, always_apply=True),
    # A.RandomBrightnessContrast(0, (0, 0.2), p=0.25),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5 for _ in range(IMG_CH)], std=[0.5 for _ in range(IMG_CH)]),
    ToTensorV2()
  ]
)
