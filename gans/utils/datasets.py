import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps

from gans.utils.helpers import walk_path

class JoinedImagePairDataset(Dataset):
  def __init__(self, root_dir, transform=None, switch_sides:bool=False, format="RGB"):
    self.format = format
    self.switch_sides = switch_sides
    self.root_dir = root_dir
    self.transform = transform
    self.files_list = walk_path(self.root_dir)
    print(f"Found {len(self.files_list)} input files")

  def __len__(self):
    return len(self.files_list)

  def __getitem__(self, index):
    img_path = self.files_list[index]
    image = Image.open(img_path)
    if self.format == "RGB":
      image = image.convert("RGB")
    else:
      image = ImageOps.grayscale(image)
    image = np.array(image)

    shape = image.shape
    width = shape[1]
    image_size = width // 2
    if not self.switch_sides:
      input_image = image[:, :image_size, :]
      target_image = image[:, image_size:, :]
    else:
      input_image = image[:, image_size:, :]
      target_image = image[:, :image_size, :]

    if self.transform is not None:
      if isinstance(self.transform, list):
        augmentations = self.transform[0](image=input_image)
        input_image = augmentations["image"]

        augmentations = self.transform[1](image=input_image)
        target_image = augmentations["image"]
      else:
        augmentations = self.transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

    return input_image, target_image

class SplitImagePairDataset(Dataset):
  def __init__(self, root: str, class_A_dir: str = "A", class_B_dir: str = "B", transform=None, format="RGB"):
    assert os.path.exists(root) and os.path.isdir(root)

    self.format = format
    self.root_A = os.path.join(root, class_A_dir)
    self.root_B = os.path.join(root, class_B_dir)
    assert os.path.exists(self.root_A) and os.path.isdir(self.root_A)
    assert os.path.exists(self.root_B) and os.path.isdir(self.root_B)

    self.transform = transform

    self.class_A_images = walk_path(self.root_A)
    self.class_B_images = walk_path(self.root_B)

    self.class_A_length = len(self.class_A_images)
    self.class_B_length = len(self.class_B_images)
    self.dataset_length = max((self.class_A_length, self.class_B_length))

  def __len__(self):
    return self.dataset_length

  def __getitem__(self, index):
    class_A_image_path = self.class_A_images[index % self.class_A_length]
    class_B_image_path = self.class_B_images[index % self.class_B_length]

    class_A_image = Image.open(class_A_image_path)
    if self.format == "RGB":
      class_A_image = class_A_image.convert("RGB")
    else:
      class_A_image = ImageOps.grayscale(class_A_image)
    class_A_image = np.array(class_A_image)

    class_B_image = Image.open(class_B_image_path)
    if self.format == "RGB":
      class_B_image = class_B_image.convert("RGB")
    else:
      class_B_image = ImageOps.grayscale(class_B_image)
    class_B_image = np.array(class_B_image)

    if self.transform is not None:
      augmentations = self.transform(image=class_A_image, image0=class_B_image)
      class_A_image, class_B_image = augmentations["image"], augmentations["image0"]

    return class_A_image, class_B_image

class SingleInTwoOutDataset(Dataset):
  def __init__(self, root_dir, both_transform=None, first_transform=None, second_transform=None, format="RGB"):
    assert os.path.exists(root_dir) and os.path.isdir(root_dir)

    self.format = format
    self.both_transform = both_transform
    self.first_transform = first_transform
    self.second_transform = second_transform
    self.image_paths = walk_path(root_dir)

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    path = self.image_paths[index]

    image = Image.open(path)
    if self.format == "RGB":
      image = image.convert("RGB")
    else:
      image = ImageOps.grayscale(image)
    image = np.array(image)

    if self.both_transform is not None:
      image = self.both_transform(image=image)["image"]

    first_image = second_image = None

    if self.first_transform is not None:
      first_image = self.first_transform(image=image)["image"]

    if self.second_transform is not None:
      second_image = self.second_transform(image=image)["image"]

    return (first_image if first_image is not None else image.copy()), (second_image if second_image is not None else image.copy())

class SingleInSingleOutDataset(Dataset):
  def __init__(self, root_dir, transform=None, format="RGB"):
    assert os.path.exists(root_dir) and os.path.isdir(root_dir)

    self.format = format
    self.transform = transform
    self.image_paths = walk_path(root_dir)

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    path = self.image_paths[index]

    image = Image.open(path)
    if self.format == "RGB":
      image = image.convert("RGB")
    else:
      image = ImageOps.grayscale(image)
    image = np.array(image)

    if self.transform is not None:
      image = self.transform(image=image)["image"]

    return image
