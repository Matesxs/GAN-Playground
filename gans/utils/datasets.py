import os
from torch.utils.data import Dataset
from pathlib import Path
import random
from multiprocessing.pool import ThreadPool

from .helpers import walk_path, load_image

class JoinedImagePairDataset(Dataset):
  def __init__(self, root_dir, transform=None, switch_sides:bool=False, format="RGB"):
    self.format = format
    self.switch_sides = switch_sides
    self.root_dir = root_dir
    self.transform = transform
    self.files_list = walk_path(self.root_dir)

  def __len__(self):
    return len(self.files_list)

  def __getitem__(self, index):
    img_path = self.files_list[index]
    image = load_image(img_path, self.format)

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
  def __init__(self, class_A_dir: str, class_B_dir: str, transform=None, format="RGB"):
    self.format = format
    assert os.path.exists(class_A_dir) and os.path.isdir(class_A_dir)
    assert os.path.exists(class_B_dir) and os.path.isdir(class_B_dir)

    self.transform = transform
    self.single_transform = not isinstance(self.transform, list) and not isinstance(self.transform, tuple)

    self.class_A_images = walk_path(class_A_dir)
    self.class_B_images = walk_path(class_B_dir)

    self.class_A_length = len(self.class_A_images)
    self.class_B_length = len(self.class_B_images)
    self.dataset_length = max((self.class_A_length, self.class_B_length))

  def __len__(self):
    return self.dataset_length

  def __getitem__(self, index):
    class_A_image_path = self.class_A_images[index % self.class_A_length]
    class_B_image_path = self.class_B_images[index % self.class_B_length]

    class_A_image = load_image(class_A_image_path, self.format)
    class_B_image = load_image(class_B_image_path, self.format)

    if self.transform is not None:
      if self.single_transform:
        augmentations = self.transform(image=class_A_image, image0=class_B_image)
        class_A_image, class_B_image = augmentations["image"], augmentations["image0"]
      else:
        class_A_image = self.transform[0](image=class_A_image)["image"]
        class_B_image = self.transform[1](image=class_B_image)["image"]

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

    image = load_image(path, self.format)

    if self.both_transform is not None:
      image = self.both_transform(image=image)["image"]

    first_image = second_image = None

    if self.first_transform is not None:
      first_image = self.first_transform(image=image)["image"]

    if self.second_transform is not None:
      second_image = self.second_transform(image=image)["image"]

    return (first_image if first_image is not None else image.copy()), (second_image if second_image is not None else image.copy())

class SOCOFingAugmentedDataset(Dataset):
  def __init__(self, true_root, augmented_root, transform, format="GRAY", only_first=False):
    assert os.path.exists(true_root) and os.path.isdir(true_root)
    assert os.path.exists(augmented_root) and os.path.isdir(augmented_root)

    self.only_first = only_first
    self.format = format
    self.transform = transform
    self.augmented_root = augmented_root

    self.true_filepaths = walk_path(true_root)
    self.true_filepaths = [Path(path) for path in self.true_filepaths]

    self.true_to_augmented_mapping = {}

    augmented_filepaths = walk_path(self.augmented_root)
    augmented_filepaths = [Path(path) for path in augmented_filepaths]
    true_files_to_delete = []

    def process_true_file_augmentation(true_file_path):
      true_filename = true_file_path.name.split(".")[0]
      tmp_files = list(filter(lambda path: true_filename in path.name.split(".")[0], augmented_filepaths))

      if tmp_files:
        self.true_to_augmented_mapping[true_file_path] = tmp_files
      else:
        true_files_to_delete.append(true_file_path)

    with ThreadPool(8) as pool:
      pool.map(process_true_file_augmentation, self.true_filepaths)

    for true_file in true_files_to_delete:
      self.true_filepaths.remove(true_file)

  def __len__(self):
    return len(self.true_filepaths)

  def __getitem__(self, index):
    true_filepath = self.true_filepaths[index]
    augmented_filepath = random.choice(self.true_to_augmented_mapping[true_filepath]) if not self.only_first else self.true_to_augmented_mapping[true_filepath][0]

    true_image = load_image(str(true_filepath), self.format)
    augmented_image = load_image(str(augmented_filepath), self.format)

    transformed_images = self.transform(image=true_image, image0=augmented_image)

    return transformed_images["image0"], transformed_images["image"]


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

    image = load_image(path, self.format)

    if self.transform is not None:
      image = self.transform(image=image)["image"]

    return image

class SingleInSingleOutWithClassDataset(Dataset):
  def __init__(self, root_dir, transform=None, format="RGB"):
    assert os.path.exists(root_dir) and os.path.isdir(root_dir)

    class_folders = []

    print("Classes:")
    for class_idx, class_name in enumerate(os.listdir(root_dir)):
      full_path = os.path.join(root_dir, class_name)
      if os.path.isdir(full_path):
        class_folders.append(full_path)
        print(f"{class_name}: {class_idx}")

    self.format = format
    self.transform = transform
    self.image_paths_to_class = {}

    for class_i, class_path in enumerate(class_folders):
      class_files = walk_path(class_path)
      for class_file in class_files:
        self.image_paths_to_class[class_file] = class_i

    self.image_paths = list(self.image_paths_to_class.keys())

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    path = self.image_paths[index]

    image = load_image(path, self.format)

    if self.transform is not None:
      image = self.transform(image=image)["image"]

    return image, self.image_paths_to_class[path]
