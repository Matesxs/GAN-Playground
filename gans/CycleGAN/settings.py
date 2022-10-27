import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "photo2monet_cycle-gan_model"

TRAIN_DIR_A = "datasets/monet/train"
TRAIN_DIR_B = "datasets/scenery_photos/train"
VAL_DIR_A = "datasets/monet/test"
VAL_DIR_B = "datasets/scenery_photos/test"

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

TRUE_LABEL_SMOOTHING = True
FAKE_LABEL_SMOOTHING = False

LR = 2e-4
DECAY_LR = True
DECAY_AFTER_ITERATIONS = 750_000 # Training iterations with full LR
DECAY_ITERATION = 750_000 # Number of iterations for which will LR decay to zero
BATCH_SIZE = 1
LAMBDA_IDENTITY = 0.6
LAMBDA_CYCLE = 10
ITERATIONS = 1_500_000

SAVE_ALWAYS_REFERENCE = False
SAMPLE_EVERY = 2_000
TESTING_SAMPLES = 16

WORKERS = 8

transforms = A.Compose(
  [
    # A.SmallestMaxSize(IMG_SIZE, interpolation=cv2.INTER_CUBIC, always_apply=True),
    # A.RandomCrop(IMG_SIZE, IMG_SIZE, always_apply=True),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5 for _ in range(IMG_CHAN)], std=[0.5 for _ in range(IMG_CHAN)], max_pixel_value=255.0, ),
    ToTensorV2()
  ],
  additional_targets={"image0":"image"}
)

test_transform = A.Compose(
  [
    # A.SmallestMaxSize(IMG_SIZE, interpolation=cv2.INTER_CUBIC, always_apply=True),
    # A.CenterCrop(IMG_SIZE, IMG_SIZE, always_apply=True),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.5 for _ in range(IMG_CHAN)], std=[0.5 for _ in range(IMG_CHAN)], max_pixel_value=255.0, ),
    ToTensorV2()
  ],
  additional_targets={"image0":"image"}
)
