import torch
import torch.nn as nn
from torchsummary import summary

from gans.utils.helpers import initialize_model
from gans.utils.global_modules import PixelShuffleConv

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample=True, use_activation=True, use_norm=True, **kwargs):
    super(ConvBlock, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, padding_mode="reflect", bias=not use_norm, **kwargs) if downsample
      else PixelShuffleConv(in_channels, scale_factor=2, bias=not use_norm, out_channels=out_channels),
      nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
      nn.ReLU(inplace=True) if use_activation else nn.Identity()
    )

  def forward(self, x):
    return self.conv(x)

class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()

    self.block = nn.Sequential(
      ConvBlock(channels, channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
      ConvBlock(channels, channels, use_activation=False, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
    )

  def forward(self, x):
    return x + self.block(x)

class Generator(nn.Module):
  def __init__(self, in_channels=3, features=64, residuals=9):
    super(Generator, self).__init__()

    self.down_layers = nn.ModuleList(
      [
        ConvBlock(in_channels, features, use_norm=False, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),

        ConvBlock(features, features * 2, downsample=True, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        ConvBlock(features * 2, features * 4, downsample=True, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      ]
    )

    self.residual_layers = nn.Sequential(*[ResidualBlock(features * 4) for _ in range(residuals)])

    self.up_layers = nn.ModuleList(
      [
        ConvBlock(features * 4, features * 2, downsample=False),
        ConvBlock(features * 2, features, downsample=False),

        nn.Conv2d(features, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), padding_mode="reflect"),
        nn.Tanh()
      ]
    )

    initialize_model(self)

  def forward(self, x):
    for layer in self.down_layers:
      x = layer(x)

    x = self.residual_layers(x)

    for layer in self.up_layers:
      x = layer(x)

    return x

if __name__ == '__main__':
  x = torch.randn((1, 3, 256, 256))

  gen = Generator()
  pred = gen(x)
  print(pred.shape)

  summary(gen, (3, 256, 256), device="cpu")

  # input_names = ['Sentence']
  # output_names = ['yhat']
  # torch.onnx.export(gen, x, 'model.onnx', input_names=input_names, output_names=output_names)
