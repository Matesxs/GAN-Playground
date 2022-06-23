import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from gans.utils.datasets import JoinedImagePairDataset, SingleInTwoOutDataset, SOCOFingAugmentedDataset
from . import settings

# Color anime
train_transform = A.Compose(
  [
    A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    #A.ColorJitter(p=0.1),
    A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,),
    ToTensorV2(),
  ],
  additional_targets={"image0": "image"},
)

test_transform = A.Compose(
  [
    A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
    A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,),
    ToTensorV2(),
  ],
  additional_targets={"image0": "image"},
)

TRAIN_DATASET = JoinedImagePairDataset(root_dir="datasets/anime/train", switch_sides=True, transform=train_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")
TEST_DATASET = JoinedImagePairDataset(root_dir="datasets/anime/val", switch_sides=True, transform=test_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")

# Gray to color
# train_both_transform = A.Compose(
#   [
#     A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.25)
#   ]
# )
#
# train_input_image_transform = A.Compose(
#   [
#     A.ToGray(p=1.0),
#     A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# train_truth_image_transform = A.Compose(
#   [
#     A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# test_both_transform = A.Compose(
#   [
#     A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE)
#   ]
# )
#
# test_input_image_transform = A.Compose(
#   [
#     A.ToGray(p=1.0),
#     A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# test_truth_image_transform = A.Compose(
#   [
#     A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# TRAIN_DATASET = SingleInTwoOutDataset(root_dir="datasets/imagenet/train", both_transform=train_both_transform, first_transform=train_input_image_transform, second_transform=train_truth_image_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")
# TEST_DATASET = SingleInTwoOutDataset(root_dir="datasets/imagenet/test", both_transform=test_both_transform, first_transform=test_input_image_transform, second_transform=test_truth_image_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")

# Fill corrupted fingerprints
# train_both_transform = A.Compose(
#   [
#     A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
#     A.HorizontalFlip(),
#     A.VerticalFlip(),
#     A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,)
#   ]
# )
#
# train_input_image_transform = A.Compose(
#   [
#     A.CoarseDropout(fill_value=1, always_apply=True, max_holes=30, max_width=50, max_height=50, min_holes=2, min_width=1, min_height=1),
#     ToTensorV2()
#   ]
# )
#
# train_truth_image_transform = A.Compose(
#   [
#     ToTensorV2()
#   ]
# )
#
# test_both_transform = A.Compose(
#   [
#     A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
#     A.Normalize(mean=[0.5 for _ in range(settings.IMG_CHAN)], std=[0.5 for _ in range(settings.IMG_CHAN)], max_pixel_value=255.0,)
#   ]
# )
#
# test_input_image_transform = A.Compose(
#   [
#     A.CoarseDropout(fill_value=1, always_apply=True, max_holes=30, max_width=50, max_height=50, min_holes=2, min_width=1, min_height=1),
#     ToTensorV2()
#   ]
# )
#
# test_truth_image_transform = A.Compose(
#   [
#     ToTensorV2()
#   ]
# )
#
# TRAIN_DATASET = SingleInTwoOutDataset(root_dir="datasets/SOCOFing/Real", both_transform=train_both_transform, first_transform=train_input_image_transform, second_transform=train_truth_image_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")
# TEST_DATASET = SingleInTwoOutDataset(root_dir="datasets/SOCOFing/Real", both_transform=test_both_transform, first_transform=test_input_image_transform, second_transform=test_truth_image_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")
