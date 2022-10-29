import torch.nn as nn
from typing import List

from gans.VQGAN.modules import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
  def __init__(self, channel_map: List[int], attention_resolutions: List[int], encoder_residuals_per_level: int, image_size: int, image_channels: int, latent_dimension: int):
    super(Encoder, self).__init__()

    layers = [nn.Conv2d(image_channels, channel_map[0], 3, 1, 1)]
    for i in range(len(channel_map) - 1):
      in_channels = channel_map[i]
      out_channels = channel_map[i + 1]

      for j in range(encoder_residuals_per_level):
        layers.append(ResidualBlock(in_channels, out_channels))
        in_channels = out_channels
        if image_size in attention_resolutions:
          layers.append(NonLocalBlock(in_channels))

      if i != len(channel_map) - 2:
        layers.append(DownSampleBlock(channel_map[i + 1]))
        image_size //= 2

    layers.append(ResidualBlock(channel_map[-1], channel_map[-1]))
    layers.append(NonLocalBlock(channel_map[-1]))
    layers.append(ResidualBlock(channel_map[-1], channel_map[-1]))
    layers.append(GroupNorm(channel_map[-1]))
    layers.append(Swish())
    layers.append(nn.Conv2d(channel_map[-1], latent_dimension, 3, 1, 1))

    self.model = nn.Sequential(*layers)
    self.final_image_size = image_size

  def get_final_resolution(self) -> int:
    return self.final_image_size

  def forward(self, x):
    return self.model(x)
