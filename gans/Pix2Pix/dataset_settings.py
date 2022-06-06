import albumentations as A
from albumentations.pytorch import ToTensorV2

from gans.utils.datasets import JoinedImagePairDataset, SingleInTwoOutDataset
import settings

train_transform = A.Compose(
  [
    A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    #A.ColorJitter(p=0.1),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),
  ],
  additional_targets={"image0": "image"},
)

test_transform = A.Compose(
  [
    A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),
  ],
  additional_targets={"image0": "image"},
)

TRAIN_DATASET = JoinedImagePairDataset(root_dir="datasets/anime/train", switch_sides=True, transform=train_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")
TEST_DATASET = JoinedImagePairDataset(root_dir="datasets/anime/val", switch_sides=True, transform=test_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")

# train_both_transform = A.Compose(
#   [
#     A.Resize(width=settings.IMG_SIZE, height=settings.IMG_SIZE),
#     A.HorizontalFlip(p=0.5),
#     A.ColorJitter(p=0.02)
#   ]
# )
#
# train_input_image_transform = A.Compose(
#   [
#     A.ToGray(),
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# train_truth_image_transform = A.Compose(
#   [
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# test_input_image_transform = A.Compose(
#   [
#     A.ToGray(),
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# test_truth_image_transform = A.Compose(
#   [
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#     ToTensorV2()
#   ]
# )
#
# TRAIN_DATASET = SingleInTwoOutDataset(root_dir="datasets/imagenet/train", both_transform=train_both_transform, first_transform=train_input_image_transform, second_transform=train_truth_image_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")
# TEST_DATASET = SingleInTwoOutDataset(root_dir="datasets/imagenet/test", first_transform=test_input_image_transform, second_transform=test_truth_image_transform, format="RGB" if settings.IMG_CHAN == 3 else "GRAY")
