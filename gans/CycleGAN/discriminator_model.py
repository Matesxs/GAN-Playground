import torch
import torch.nn as nn
from torchsummary import summary

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride, use_norm=True):
    super(Block, self).__init__()

    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=stride, padding=(1, 1), bias=True, padding_mode="reflect")]
    if use_norm:
      nn.InstanceNorm2d(out_channels)
    layers.append(nn.LeakyReLU(0.2))

    self.conv = nn.Sequential(*layers)

  def forward(self, x):
    return self.conv(x)

class Discriminator(nn.Module):
  def __init__(self, channels=3, features=None, strides=None):
    super(Discriminator, self).__init__()

    if strides is None:
      strides = [(2, 2), (2, 2), (2, 2), (1, 1)]

    if features is None:
      features = [64, 128, 256, 512]

    layers = []
    for idx, (feature, stride) in enumerate(zip(features, strides)):
      layers.append(Block(channels, feature, stride=stride, use_norm=False if idx == 0 else True))
      channels = feature

    layers.append(nn.Conv2d(channels, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), padding_mode="reflect"))
    layers.append(nn.Sigmoid())

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

if __name__ == '__main__':
  x = torch.randn((1, 3, 256, 256))

  disc = Discriminator()
  preds = disc(x)
  print(preds.shape)

  summary(disc, (3, 256, 256), device="cpu")
