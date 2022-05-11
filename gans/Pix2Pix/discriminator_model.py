import torch
import torch.nn as nn
from torchsummary import summary

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=(2,2)):
    super(CNNBlock, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, (4, 4), stride, bias=False, padding=1, padding_mode="reflect"),
      nn.InstanceNorm2d(out_channels, affine=True),
      nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.conv(x)

class Discriminator(nn.Module):
  def __init__(self, channels=3, features=None, strides=None):
    super(Discriminator, self).__init__()

    if strides is None:
      strides = [(2, 2), (2, 2), (2, 2), (1, 1)]
    
    if features is None:
      features = [64, 128, 256, 512]

    self.initial = nn.Sequential(
      nn.Conv2d(channels * 2, features[0], kernel_size=(4, 4), stride=strides[0], padding=1, padding_mode="reflect"),
      nn.LeakyReLU(0.2)
    )

    layers = []
    channels = features[0]
    for feature, stride in zip(features[1:], strides[1:]):
      layers.append(CNNBlock(channels, feature, stride=stride))
      channels = feature

    layers.append(nn.Conv2d(channels, 1, kernel_size=(4, 4), stride=(1, 1), padding=1, padding_mode="reflect"))

    self.model = nn.Sequential(*layers)

  def forward(self, x, y):
    x = torch.cat([x, y], dim=1)
    x = self.initial(x)
    return self.model(x)

if __name__ == '__main__':
  x = torch.randn((1, 3, 256, 256))
  y = torch.randn((1, 3, 256, 256))

  disc = Discriminator()
  preds = disc(x, y)
  print(preds.shape)

  summary(disc, [(3, 256, 256), (3, 256, 256)], device="cpu")
