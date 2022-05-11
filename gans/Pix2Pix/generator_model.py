import torch
import torch.nn as nn
from torchsummary import summary

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, downscale=True, activation="relu", dropout=False):
    super(Block, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False, padding_mode="reflect")
      if downscale
      else nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
      nn.InstanceNorm2d(out_channels, affine=True),
      nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)
    )

    self.use_dropout = dropout
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.conv(x)
    if self.use_dropout:
      x = self.dropout(x)
    return x

class Generator(nn.Module):
  def __init__(self, in_channels=3, features=None, dropouts_up=None):
    super(Generator, self).__init__()

    if features is None:
      features = [64, 128, 256, 512, 512, 512, 512]

    if dropouts_up is None or len(features) != len(dropouts_up):
      dropouts_up = [True, True, True, False, False, False, False]

    self.initial = nn.Sequential(
      nn.Conv2d(in_channels, features[0], (4, 4), (2, 2), (1, 1), padding_mode="reflect"),
      nn.LeakyReLU(0.2)
    )

    channels = features[0]
    self.down_layers = nn.ModuleList()
    for feature in features[1:]:
      self.down_layers.append(Block(channels, feature, downscale=True, activation="leaky", dropout=False))
      channels = feature

    self.bottleneck = nn.Sequential(
      nn.Conv2d(channels, channels, (4, 4), (2, 2), (1, 1), padding_mode="reflect"),
      nn.ReLU()
    )

    features = list(reversed(features))
    self.initial_up = Block(channels, features[0], downscale=False, activation="relu", dropout=dropouts_up[0])
    channels = features[0]

    self.up_layers = nn.ModuleList()
    for feature, dropout in zip(features[1:], dropouts_up[1:]):
      self.up_layers.append(Block(channels * 2, feature, downscale=False, activation="relu", dropout=dropout))
      channels = feature

    self.final_up = nn.Sequential(
      nn.ConvTranspose2d(channels * 2, in_channels, (4, 4), (2, 2), (1, 1)),
      nn.Tanh()
    )

  def forward(self, x):
    out = initial_down = self.initial(x)

    down_skips = []
    for down_layer in self.down_layers:
      out = down_layer(out)
      down_skips.append(out)

    out = self.bottleneck(out)
    out = self.initial_up(out)

    down_skips = list(reversed(down_skips))
    for up_layer, skip in zip(self.up_layers, down_skips):
      out = up_layer(torch.cat([out, skip], dim=1))

    return self.final_up(torch.cat([out, initial_down], dim=1))

if __name__ == '__main__':
  x = torch.randn((1, 3, 256, 256))

  gen = Generator()
  pred = gen(x)
  print(pred.shape)

  summary(gen, (3, 256, 256), device="cpu")
