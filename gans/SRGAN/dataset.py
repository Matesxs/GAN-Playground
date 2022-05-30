import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import settings
from gans.utils.helpers import walk_path

class SrganDataset(Dataset):
  def __init__(self, root_dir, transforms=None):
    assert os.path.exists(root_dir) and os.path.isdir(root_dir)

    self.transforms = transforms
    self.image_paths = walk_path(root_dir)

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    path = self.image_paths[index]

    image = np.array(Image.open(path).convert("RGB"))

    if self.transforms is None:
      image = settings.both_transform(image=image)["image"]
      high_res = settings.high_res_transform(image=image)["image"]
      low_res = settings.low_res_transform(image=image)["image"]
      return low_res, high_res
    else:
      transforms = [transform(image=image)["image"] for transform in self.transforms]
      return transforms

if __name__ == '__main__':
  dataset = SrganDataset("datasets/imagenet/test", transforms=[settings.test_transform])
  loader = DataLoader(dataset, 16, pin_memory=True)
  data = next(iter(loader))
  print(data[0].shape)
