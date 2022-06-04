import torch
import torch.nn as nn

class PixelNorm(nn.Module):
  def __init__(self, epsilon=1e-8):
    super(PixelNorm, self).__init__()

    self.epsilon = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class PixelShuffleConv(nn.Module):
  def __init__(self, in_channels, scale_factor=2, kernel=3, padding=1, out_channels=None):
    super(PixelShuffleConv, self).__init__()
    if out_channels is None:
      out_channels = in_channels

    self.block = nn.Sequential(
      nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel, 1, padding),
      nn.PixelShuffle(scale_factor) if scale_factor > 1 else nn.Identity()
    )

  def forward(self, x):
    return self.block(x)