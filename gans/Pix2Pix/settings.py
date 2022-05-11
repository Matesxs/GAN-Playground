import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "anime_pix2pix_model"

SWITCH_INPUT_IMAGE_POSITIONS = True
TESTING_SAMPLES = 8
TRAINING_DATASET_PATH = "datasets/anime/train"
TESTING_DATASET_PATH = "datasets/anime/val"

GEN_MODEL_WEIGHTS_TO_LOAD = None
DISC_MODEL_WEIGHTS_TO_LOAD = None

LR = 2e-4
BATCH_SIZE = 16
IMG_SIZE = 256
IMG_CHAN = 3
FEATURES_DISC = [64, 128, 256, 512] # [64 * 2, 128 * 2, 256 * 2, 512 * 2] [64, 128, 256, 512]
FEATURES_GEN = [64, 128, 256, 512, 512, 512, 512] # [64 * 2, 128 * 2, 256 * 2, 512 * 2, 512 * 2, 512 * 2, 512 * 2] [64, 128, 256, 512, 512, 512, 512]
GEN_UPSCALE_DROPOUT = [True, True, True, False, False, False, False]
L1_LAMBDA = 100
EPOCHS = 1000

both_transform = A.Compose(
    [A.Resize(width=IMG_SIZE, height=IMG_SIZE), A.HorizontalFlip(p=0.5),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        #A.ColorJitter(p=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
