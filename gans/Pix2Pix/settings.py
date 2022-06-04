import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "anime_pix2pix_model"

IMG_SIZE = 256
IMG_CHAN = 3

SWITCH_INPUT_IMAGE_POSITIONS = True
TRAINING_DATASET_PATH = "datasets/anime/train"
TESTING_DATASET_PATH = "datasets/anime/val"

GEN_MODEL_WEIGHTS_TO_LOAD = None
DISC_MODEL_WEIGHTS_TO_LOAD = None

SAVE_CHECKPOINT = False
CHECKPOINT_EVERY = 5_000

TESTING_SAMPLES = 8
SAMPLE_EVERY = 1_000

ITERATIONS = 1_000_000
BATCH_SIZE = 8
LR = 2e-4
L1_LAMBDA = 100
GP_LAMBDA = 0
FEATURES_DISC = [64, 128, 256, 512] # [128, 256, 512, 1024] [64, 128, 256, 512]
FEATURES_GEN = [64, 128, 256, 512, 512, 512, 512] # [64, 128, 256, 512, 1024, 1024, 1024] [64, 128, 256, 512, 512, 512, 512]
GEN_UPSCALE_DROPOUT = [True, True, True, False, False, False, False]
TRUE_LABEL_SMOOTHING = False
FAKE_LABEL_SMOOTHING = True

train_transform = A.Compose(
  [
    A.Resize(width=IMG_SIZE, height=IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    #A.ColorJitter(p=0.1),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),
  ],
  additional_targets={"image0": "image"},
)

test_transform = A.Compose(
  [
    A.Resize(width=IMG_SIZE, height=IMG_SIZE),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),
  ],
  additional_targets={"image0": "image"},
)

NUM_OF_WORKERS = 4
