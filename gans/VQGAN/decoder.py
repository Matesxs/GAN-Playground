import torch.nn as nn
from typing import List

from gans.VQGAN.modules import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish


class Decoder(nn.Module):
  def __init__(self, channel_map: List[int], attention_resolutions: List[int], residuals_per_level: int, start_dimension: int, image_channels: int, latent_dimension: int):
    super(Decoder, self).__init__()

    in_channels = channel_map[0]
    layers = [nn.Conv2d(latent_dimension, in_channels, 3, 1, 1),
              ResidualBlock(in_channels, in_channels),
              NonLocalBlock(in_channels),
              ResidualBlock(in_channels, in_channels)]

    for i in range(len(channel_map)):
      out_channels = channel_map[i]

      for j in range(residuals_per_level):
        layers.append(ResidualBlock(in_channels, out_channels))
        in_channels = out_channels

        if start_dimension in attention_resolutions:
          layers.append(NonLocalBlock(in_channels))

      if i != 0:
        layers.append(UpSampleBlock(in_channels))
        start_dimension *= 2

    layers.append(GroupNorm(in_channels))
    layers.append(Swish())
    layers.append(nn.Conv2d(in_channels, image_channels, 3, 1, 1))

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)
