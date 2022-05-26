import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "imagenet_srgan_model"

DATASET_PATH = "datasets/imagenet/train"
TEST_DATASET_PATH = "datasets/imagenet/test"

TEST_IMG_SIZE = 64
HIGH_RES_IMG_SIZE = 96
LOW_RES_IMG_SIZE = HIGH_RES_IMG_SIZE // 4
IMG_CH = 3

GEN_MODEL_WEIGHTS_TO_LOAD = None
DISC_MODEL_WEIGHTS_TO_LOAD = None

SAVE_CHECKPOINTS = True
CHECKPOINT_EVERY = 5

FAKE_LABEL_SMOOTHING = False
TRUE_LABEL_SMOOTHING = True

SAMPLE_EVERY = 1
TESTING_SAMPLES = 16

GEN_FEATURES = 64
DISC_FEATURES = [64, 64, 128, 128, 256, 256, 512, 512]

LR = 1e-4
GP_LAMBDA = 0
PRETRAIN_EPOCHS = 150
EPOCHS = 20
BATCH_SIZE = 16

NUM_OF_WORKERS = 8

test_transform = A.Compose(
  [
    A.Resize(width=TEST_IMG_SIZE, height=TEST_IMG_SIZE, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
  ]
)

both_transform = A.Compose(
  [
    A.RandomCrop(width=HIGH_RES_IMG_SIZE, height=HIGH_RES_IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5)
  ]
)

high_res_transform = A.Compose(
  [
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
  ]
)

low_res_transform = A.Compose(
  [
    A.Resize(width=LOW_RES_IMG_SIZE, height=LOW_RES_IMG_SIZE, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
  ]
)
