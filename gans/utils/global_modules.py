import torch
import torch.nn as nn
import numpy as np

class PixelNorm(nn.Module):
  def __init__(self, epsilon=1e-8):
    super(PixelNorm, self).__init__()

    self.epsilon = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class PixelShuffleConv(nn.Module):
  def __init__(self, in_channels, scale_factor=2, kernel=3, padding=1, out_channels=None, bias=True):
    super(PixelShuffleConv, self).__init__()
    if out_channels is None:
      out_channels = in_channels

    self.block = nn.Sequential(
      nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel, 1, padding, bias=bias),
      nn.PixelShuffle(scale_factor) if scale_factor > 1 else nn.Identity()
    )

  def forward(self, x):
    return self.block(x)

class ReshapeLayer(nn.Module):
  def __init__(self, shape):
    super(ReshapeLayer, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(-1, *self.shape)

class WeightedScaleConv2(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=np.sqrt(2)):
    super(WeightedScaleConv2, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    self.bias = self.conv.bias
    self.conv.bias = None

    convShape = list(self.conv.weight.shape)
    fanIn = np.prod(convShape[1:])
    self.wtScale = gain / np.sqrt(fanIn)

    nn.init.normal_(self.conv.weight)
    nn.init.constant_(self.bias, val=0)

  def forward(self, x):
    return self.conv(x) * self.wtScale + self.bias.view(1, self.bias.shape[0], 1, 1)