import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "monet2photo_cycle-gan_model"

TRAIN_DIR = "datasets/monet2photo/train"
VAL_DIR = "datasets/monet2photo/val"

IMG_SIZE = 256
IMG_CHAN = 3

GEN_A_MODEL_WEIGHTS_TO_LOAD = None
GEN_B_MODEL_WEIGHTS_TO_LOAD = None
DISC_A_MODEL_WEIGHTS_TO_LOAD = None
DISC_B_MODEL_WEIGHTS_TO_LOAD = None

SAVE_CHECKPOINTS = True
CHECKPOINT_EVERY = 5_000

FEATURES_DISC = [64, 128, 256, 512]
FEATURES_GEN = 64
GEN_RESIDUAL = 9

LR = 2e-4
BATCH_SIZE = 1
LAMBDA_IDENTITY = 5
LAMBDA_CYCLE = 10
ITERATIONS = 200_000

SAMPLE_EVERY = 1_000
TESTING_SAMPLES = 4

WORKERS = 4

transforms = A.Compose(
  [
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
    ToTensorV2()
  ],
  additional_targets={"image0":"image"}
)

test_transform = A.Compose(
  [
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
    ToTensorV2()
  ],
  additional_targets={"image0":"image"}
)
