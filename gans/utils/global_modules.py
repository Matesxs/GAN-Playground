import torch
import torch.nn as nn

class PixelNorm(nn.Module):
  def __init__(self, epsilon=1e-8):
    super(PixelNorm, self).__init__()

    self.epsilon = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class PixelShuffleConv(nn.Module):
  def __init__(self, in_channels, scale_factor=2, kernel=3, padding=1):
    super(PixelShuffleConv, self).__init__()

    self.block = nn.Sequential(
      nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel, padding, 1),
      nn.PixelShuffle(scale_factor)
    )

  def forward(self, x):
    return self.block(x)