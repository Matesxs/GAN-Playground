import torch
import torch.nn as nn

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
  def __init__(self, in_channels=3, features=64):
    super(Generator, self).__init__()

    self.initial = nn.Sequential(
      nn.Conv2d(in_channels, features, (4, 4), (2, 2), (1, 1), padding_mode="reflect"),
      nn.LeakyReLU(0.2)
    )

    self.down1 = Block(features,     features * 2, downscale=True, activation="leaky", dropout=False)
    self.down2 = Block(features * 2, features * 4, downscale=True, activation="leaky", dropout=False)
    self.down3 = Block(features * 4, features * 8, downscale=True, activation="leaky", dropout=False)
    self.down4 = Block(features * 8, features * 8, downscale=True, activation="leaky", dropout=False)
    self.down5 = Block(features * 8, features * 8, downscale=True, activation="leaky", dropout=False)
    self.down6 = Block(features * 8, features * 8, downscale=True, activation="leaky", dropout=False)
    self.bottleneck = nn.Sequential(
      nn.Conv2d(features * 8, features * 8, (4, 4), (2, 2), (1, 1), padding_mode="reflect"),
      nn.ReLU()
    )
    self.up1 = Block(features * 8, features * 8, downscale=False, activation="relu", dropout=True)
    self.up2 = Block(features * 8 * 2, features * 8, downscale=False, activation="relu", dropout=True)
    self.up3 = Block(features * 8 * 2, features * 8, downscale=False, activation="relu", dropout=True)
    self.up4 = Block(features * 8 * 2, features * 8, downscale=False, activation="relu", dropout=True)
    self.up5 = Block(features * 8 * 2, features * 4, downscale=False, activation="relu", dropout=True)
    self.up6 = Block(features * 4 * 2, features * 2, downscale=False, activation="relu", dropout=True)
    self.up7 = Block(features * 2 * 2, features, downscale=False, activation="relu", dropout=True)
    self.final_up = nn.Sequential(
      nn.ConvTranspose2d(features * 2, in_channels, (4, 4), (2, 2), (1, 1)),
      nn.Tanh()
    )

  def forward(self, x):
    d1 = self.initial(x)
    d2 = self.down1(d1)
    d3 = self.down2(d2)
    d4 = self.down3(d3)
    d5 = self.down4(d4)
    d6 = self.down5(d5)
    d7 = self.down6(d6)
    bottleneck = self.bottleneck(d7)
    up1 = self.up1(bottleneck)
    up2 = self.up2(torch.cat([up1, d7], dim=1))
    up3 = self.up3(torch.cat([up2, d6], dim=1))
    up4 = self.up4(torch.cat([up3, d5], dim=1))
    up5 = self.up5(torch.cat([up4, d4], dim=1))
    up6 = self.up6(torch.cat([up5, d3], dim=1))
    up7 = self.up7(torch.cat([up6, d2], dim=1))
    return self.final_up(torch.cat([up7, d1], dim=1))

if __name__ == '__main__':
  x = torch.randn((1, 3, 256, 256))

  gen = Generator()
  pred = gen(x)
  print(pred.shape)
