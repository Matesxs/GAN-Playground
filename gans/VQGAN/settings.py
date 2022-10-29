import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "monet_vqgan_model"
DATASET_PATH = "datasets/monet/train"
VAL_DATASET_PATH = "datasets/monet/test"
IMG_SIZE = 256
IMG_CH = 3

# Reconstructor (Encoder/Decoder)
ENCODER_FILTERS = [128, 128, 128, 256, 256, 512]
DECODER_FILTERS = [512, 256, 256, 128, 128]
ATTENTION_RESOLUTIONS = [16]
ENCODER_RESIDUAL_PER_LEVEL = 2
DECODER_RESIDUAL_PER_LEVEL = 3
LATENT_DIMENSION = 256
NUMBER_OF_LATENT_VECTORS = 1024
BETA = 0.25

# Discriminator
DISC_BASE_FILTERS = 64
DISC_MAX_FILTER_MULTIPLIER = 8
DISC_LAYERS = 3

# Training
LR = 1e-4
OPT_BETA1 = 0.5
OPT_BETA2 = 0.9
BATCH_SIZE = 2
EPOCHS = 200
DISC_TRAINING_DELAY_ITERATIONS = 5_000
DISC_FACTOR = 1.0
PERCEPTUAL_LOSS_FACTOR = 1.0
RECONSTRUCTION_LOSS_FACTOR = 1.0

CHECKPOINT_EVERY = 2_000
SAMPLE_EVERY = 500
NUMBER_OF_SAMPLES = 16

NUM_OF_WORKERS = 8

training_transform = A.Compose(
  [
    A.SmallestMaxSize(IMG_SIZE, interpolation=cv2.INTER_CUBIC, always_apply=True),
    A.CenterCrop(IMG_SIZE, IMG_SIZE, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5 for _ in range(IMG_CH)], std=[0.5 for _ in range(IMG_CH)]),
    ToTensorV2()
  ]
)

eval_transform = A.Compose(
  [
    A.SmallestMaxSize(IMG_SIZE, interpolation=cv2.INTER_CUBIC, always_apply=True),
    A.CenterCrop(IMG_SIZE, IMG_SIZE, always_apply=True),
    A.Normalize(mean=[0.5 for _ in range(IMG_CH)], std=[0.5 for _ in range(IMG_CH)]),
    ToTensorV2()
  ]
)
