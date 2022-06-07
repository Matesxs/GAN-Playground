import os
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

def walk_path(root):
  output_files = []
  for currentpath, folders, files in os.walk(root):
    for file in files:
      output_files.append(os.path.join(currentpath, file))
  return output_files

def load_image(image_path, format="RGB"):
  image = Image.open(image_path)
  if format == "RGB":
    image = image.convert("RGB")
  else:
    image = ImageOps.grayscale(image)
  return np.array(image)

def initialize_model(model:nn.Module):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)