from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import settings

class PairDataset(Dataset):
  def __init__(self, root_dir, switch_sides:bool=False):
    self.switch_sides = switch_sides
    self.root_dir = root_dir
    self.files_list = [os.path.join(self.root_dir, fil_name) for fil_name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, fil_name))]
    print(f"Found {len(self.files_list)} input files")

  def __len__(self):
    return len(self.files_list)

  def __getitem__(self, index):
    img_path = self.files_list[index]
    image = np.array(Image.open(img_path))

    shape = image.shape
    width = shape[1]
    image_size = width // 2
    if not self.switch_sides:
      input_image = image[:, :image_size, :]
      target_image = image[:, image_size:, :]
    else:
      input_image = image[:, image_size:, :]
      target_image = image[:, :image_size, :]

    augmentations = settings.both_transform(image=input_image, image0=target_image)
    input_image, target_image = augmentations["image"], augmentations["image0"]

    input_image = settings.transform_only_input(image=input_image)["image"]
    target_image = settings.transform_only_mask(image=target_image)["image"]

    return input_image, target_image
