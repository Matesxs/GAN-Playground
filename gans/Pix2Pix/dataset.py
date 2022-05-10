from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import settings

class PairDataset(Dataset):
  def __init__(self, root_dir, image_size=600):
    self.image_size = image_size
    self.root_dir = root_dir
    self.files_list = [os.path.join(self.root_dir, fil_name) for fil_name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, fil_name))]
    print(f"Found {len(self.files_list)} input files")

  def __len__(self):
    return len(self.files_list)

  def __getitem__(self, index):
    img_path = self.files_list[index]
    image = np.array(Image.open(img_path))
    input_image = image[:, :self.image_size, :]
    target_image = image[:, self.image_size:, :]

    augmentations = settings.both_transform(image=input_image, image0=target_image)
    input_image, target_image = augmentations["image"], augmentations["image0"]

    input_image = settings.transform_only_input(image=input_image)["image"]
    target_image = settings.transform_only_mask(image=target_image)["image"]

    return input_image, target_image
