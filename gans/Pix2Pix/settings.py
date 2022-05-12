import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "maps_pix2pix_model"

SWITCH_INPUT_IMAGE_POSITIONS = False
TESTING_SAMPLES = 8
TRAINING_DATASET_PATH = "datasets/maps/train"
TESTING_DATASET_PATH = "datasets/maps/val"

GEN_MODEL_WEIGHTS_TO_LOAD = None
DISC_MODEL_WEIGHTS_TO_LOAD = None

LR = 2e-4
LR_DECAY = False
LR_DECAY_COEF = 0.1
LR_DECAY_EVERY = 50

BATCH_SIZE = 1
IMG_SIZE = 256
IMG_CHAN = 3
FEATURES_DISC = [64, 128, 256, 512] # [64 * 2, 128 * 2, 256 * 2, 512 * 2] [64, 128, 256, 512]
FEATURES_GEN = [64, 128, 256, 512, 512, 512, 512] # [64 * 2, 128 * 2, 256 * 2, 512 * 2, 512 * 2, 512 * 2, 512 * 2] [64, 128, 256, 512, 512, 512, 512]
GEN_UPSCALE_DROPOUT = [True, True, True, False, False, False, False]
L1_LAMBDA = 100
EPOCHS = 200

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
