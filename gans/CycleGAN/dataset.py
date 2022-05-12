from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ImagePairDataset(Dataset):
  def __init__(self, root:str, class_A_dir:str="A", class_B_dir:str="B", transform=None):
    self.root_A = os.path.join(root, class_A_dir)
    self.root_B = os.path.join(root, class_B_dir)
    self.transform = transform

    self.class_A_images = [os.path.join(self.root_A, fn) for fn in os.listdir(self.root_A) if os.path.isfile(os.path.join(self.root_A, fn))]
    self.class_B_images = [os.path.join(self.root_B, fn) for fn in os.listdir(self.root_B) if os.path.isfile(os.path.join(self.root_B, fn))]

    self.class_A_length = len(self.class_A_images)
    self.class_B_length = len(self.class_B_images)
    self.dataset_length = max((self.class_A_length, self.class_B_length))

  def __len__(self):
    return self.dataset_length

  def __getitem__(self, index):
    class_A_image_path = self.class_A_images[index % self.class_A_length]
    class_B_image_path = self.class_B_images[index % self.class_B_length]

    class_A_image = np.array(Image.open(class_A_image_path).convert("RGB"))
    class_B_image = np.array(Image.open(class_B_image_path).convert("RGB"))

    if self.transform is not None:
      augmentations = self.transform(image=class_A_image, image0=class_B_image)
      class_A_image, class_B_image = augmentations["image"], augmentations["image0"]

    return class_A_image, class_B_image