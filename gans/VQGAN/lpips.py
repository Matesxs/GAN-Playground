import os
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple
import requests
from tqdm import tqdm

def download():
  os.makedirs("vgg_lpips", exist_ok=True)
  with requests.get("https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1", stream=True) as r:
    total_size = int(r.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
      with open("vgg_lpips/vgg.pth", "wb") as f:
        for data in r.iter_content(chunk_size=1024):
          if data:
            f.write(data)
            pbar.update(1024)

def check_ckpt_path():
  if not os.path.exists("vgg_lpips/vgg.pth"):
    print(f"Downloading VGG model")
    download()

class LPIPS(nn.Module):
  def __init__(self):
    super(LPIPS, self).__init__()

    self.scaling_layer = ScalingLayer()
    self.channels = [64, 128, 256, 512, 512]
    self.vgg = VGG16()
    self.lins = nn.ModuleList([
      NetLinLayer(channels) for channels in self.channels
    ])

    self.load_from_pretrained()

    for param in self.parameters():
      param.requires_grad = False

  def load_from_pretrained(self):
    check_ckpt_path()
    self.load_state_dict(torch.load("vgg_lpips/vgg.pth", map_location=torch.device("cpu")), strict=False)
    
  def forward(self, real_x, fake_x):
    features_real = self.vgg(self.scaling_layer(real_x))
    features_fake = self.vgg(self.scaling_layer(fake_x))
    diffs = {}

    for i in range(len(self.channels)):
      diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2

    return sum([spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.channels))])

class ScalingLayer(nn.Module):
  def __init__(self):
    super(ScalingLayer, self).__init__()
    self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
    self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

  def forward(self, x):
    return (x - self.shift) / self.scale

class NetLinLayer(nn.Module):
  def __init__(self, in_channels: int, out_channels: int=1):
    super(NetLinLayer, self).__init__()

    self.model = nn.Sequential(
      nn.Dropout(),
      nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    )
    
class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()

    vgg_pretrained_features = vgg16(pretrained=True).features
    slices = [vgg_pretrained_features[i] for i in range(30)]

    self.slice1 = nn.Sequential(*slices[0:4])
    self.slice2 = nn.Sequential(*slices[4:9])
    self.slice3 = nn.Sequential(*slices[9:16])
    self.slice4 = nn.Sequential(*slices[16:23])
    self.slice5 = nn.Sequential(*slices[23:30])

    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    h = self.slice1(x)
    h_relu1 = h
    h = self.slice2(h)
    h_relu2 = h
    h = self.slice3(h)
    h_relu3 = h
    h = self.slice4(h)
    h_relu4 = h
    h = self.slice5(h)
    h_relu5 = h

    vgg_outputs = namedtuple("VGGOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
    return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

def norm_tensor(x):
  norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
  return x / (norm_factor + 1e-10)

def spatial_average(x):
  return x.mean([2, 3], keepdim=True)
