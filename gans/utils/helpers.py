import os
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
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

def switchTrainable(nNet, status):
  for p in nNet.parameters(): p.requires_grad = status

def initialize_model(model:nn.Module):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
      if m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def inception_score(img_batches, batch_size, resize=False, splits=1):
  """Computes the inception score of the generated images imgs
  img_batches -- Array image batches
  splits -- number of splits
  """
  numbner_of_batches = len(img_batches)
  assert numbner_of_batches != 0
  assert batch_size > 0
  N = numbner_of_batches * batch_size

  # Set up dtype
  if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
  else:
    dtype = torch.FloatTensor

  # Load inception model
  inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
  inception_model.eval()
  up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

  def get_pred(x):
    if resize:
      x = up(x)
    x = inception_model(x)
    return F.softmax(x, dim=0).data.cpu().numpy()

  # Get predictions
  preds = np.zeros((N, 1000))
  for i, batch in enumerate(img_batches):
    batchv = Variable(batch)
    batch_size_i = batch.size()[0]

    preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

  # Now compute the mean kl-div
  split_scores = []

  for k in range(splits):
    part = preds[k * (N // splits): (k + 1) * (N // splits), :]
    py = np.mean(part, axis=0)
    scores = []
    for i in range(part.shape[0]):
      pyx = part[i, :]
      scores.append(entropy(pyx, py))
    split_scores.append(np.exp(np.mean(scores)))

  return np.mean(split_scores), np.std(split_scores)