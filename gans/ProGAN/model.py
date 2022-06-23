import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

from gans.utils.global_modules import PixelNorm, ReshapeLayer, WeightedScaleConv2

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, padding, stride=1, use_pixelnorm=True):
    super(ConvBlock, self).__init__()

    self.block = nn.Sequential(
      WeightedScaleConv2(in_channels, out_channels, kernel, stride, padding),
      nn.LeakyReLU(0.2),
      PixelNorm() if use_pixelnorm else nn.Identity()
    )

  def forward(self, x):
    return self.block(x)

class ToRGBBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
    super(ToRGBBlock, self).__init__()

    self.block = WeightedScaleConv2(in_channels, out_channels, kernel, stride, padding, gain=1)

  def forward(self, x):
    return self.block(x)

class FromRGBBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
    super(FromRGBBlock, self).__init__()

    self.block = nn.Sequential(
      WeightedScaleConv2(in_channels, out_channels, kernel, stride, padding),
      nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.block(x)

class BatchStdConcat(nn.Module):
  def __init__(self, groupSize=4):
    super().__init__()
    self.groupSize = groupSize

  def forward(self, x):
    shape = list(x.size())  # NCHW - Initial size
    actual_group_size = self.groupSize if self.groupSize < shape[0] else shape[0] # Attempt to get expected group size or maximum available
    xStd = x.view(actual_group_size, -1, shape[1], shape[2], shape[3])
    xStd -= torch.mean(xStd, dim=0, keepdim=True)  # GMCHW - Subract mean of shape 1MCHW
    xStd = torch.mean(xStd ** 2, dim=0, keepdim=False)  # MCHW - Take mean of squares
    xStd = (xStd + 1e-08) ** 0.5  # MCHW - Take std
    xStd = torch.mean(xStd.view(int(shape[0] / actual_group_size), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
    # M111 - Take mean across CHW
    xStd = xStd.repeat(actual_group_size, 1, shape[2], shape[3])  # N1HW - Expand to same shape as x with one channel
    return torch.cat([x, xStd], 1)

class Generator(nn.Module):
  def __init__(self, img_channels=3, target_resolution=256, feature_base=8192, feature_max=512, z_dim=512):
    super(Generator, self).__init__()

    self.feature_base = feature_base
    self.feature_max = feature_max
    self.number_of_blocks = int(np.log2(target_resolution))

    self.blocks = nn.ModuleList()
    self.rgb_blocks = nn.ModuleList()

    # Initial block
    in_channels, out_channels = z_dim, self.get_number_of_filters(1)
    initial_block = nn.Sequential(
      ReshapeLayer([z_dim, 1, 1]),
      ConvBlock(in_channels, out_channels, kernel=4, padding=3),
      ConvBlock(out_channels, out_channels, kernel=3, padding=1)
    )
    toRGB = ToRGBBlock(out_channels, img_channels)
    self.blocks.append(initial_block)
    self.rgb_blocks.append(toRGB)

    for i in range(2, self.number_of_blocks):
      in_channels, out_channels = self.get_number_of_filters(i - 1), self.get_number_of_filters(i)
      block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        ConvBlock(in_channels, out_channels, kernel=3, padding=1),
        ConvBlock(out_channels, out_channels, kernel=3, padding=1)
      )
      toRGB = ToRGBBlock(out_channels, img_channels)

      self.blocks.append(block)
      self.rgb_blocks.append(toRGB)

  def get_number_of_filters(self, stage):
    return min(int(self.feature_base / (2.0 ** stage)), self.feature_max)

  def forward(self, x, alpha=1.0, stage=None):
    if stage is None:
      stage = self.number_of_blocks - 2

    prev_stage = stage - 1
    fade = alpha < 1.0

    for lev in range(stage + 1):
      x = self.blocks[lev](x)

      if lev == prev_stage and fade:
        prevLevel_x = self.rgb_blocks[lev](x)

      if lev == stage:
        x = self.rgb_blocks[lev](x)

        # Fade only when alpha is less than 1.0 and we are not in first stage
        if fade and stage != 0:
          x = alpha * x + (1 - alpha) * F.interpolate(prevLevel_x, scale_factor=2, mode='nearest')
    return torch.tanh(x)

class Critic(nn.Module):
  def __init__(self, img_channels=3, target_resolution=256, feature_base=8192, feature_max=512):
    super(Critic, self).__init__()

    self.feature_base = feature_base
    self.feature_max = feature_max
    self.number_of_blocks = int(np.log2(target_resolution))

    self.blocks = nn.ModuleList()
    self.from_rgb_blocks = nn.ModuleList()

    # last preprocessing layer
    in_channels, out_channels = img_channels, self.get_number_of_filters(self.number_of_blocks - 1)
    self.from_rgb_blocks.append(FromRGBBlock(in_channels, out_channels))

    for i in range(self.number_of_blocks - 1, 1, -1):
      in_channels, out_channels = self.get_number_of_filters(i), self.get_number_of_filters(i - 1)
      block = nn.Sequential(
        ConvBlock(in_channels, in_channels, kernel=3, padding=1, use_pixelnorm=False),
        ConvBlock(in_channels, out_channels, kernel=3, padding=1, use_pixelnorm=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
      )
      fromRGB = FromRGBBlock(img_channels, out_channels)

      self.blocks.append(block)
      self.from_rgb_blocks.append(fromRGB)

    first_block = nn.ModuleList()

    in_channels, out_channels = self.get_number_of_filters(1), self.get_number_of_filters(1)
    first_block.append(BatchStdConcat())
    in_channels += 1

    first_block.append(ConvBlock(in_channels, out_channels, kernel=3, padding=1, use_pixelnorm=False))
    in_channels, out_channels = out_channels, self.get_number_of_filters(0)

    first_block.append(ConvBlock(in_channels, out_channels, kernel=4, padding=0, use_pixelnorm=False))
    in_channels, out_channels = out_channels, 1

    first_block.append(WeightedScaleConv2(in_channels, out_channels, kernel_size=1, stride=1, padding=0, gain=1))

    self.blocks.append(nn.Sequential(*first_block))

  def get_number_of_filters(self, stage):
    return min(int(self.feature_base / (2.0 ** stage)), self.feature_max)

  def forward(self, x, alpha=1.0, stage=None):
    levels = len(self.blocks)
    if stage is None:
      stage = self.number_of_blocks - 2

    fade = alpha < 1.0 and stage != 0
    current_stage = levels - stage - 1
    prev_stage = current_stage + (1 if fade else 0)

    if not fade:
      x = self.from_rgb_blocks[current_stage](x)
      x = self.blocks[current_stage](x)
    else:
      curLevel_x = self.from_rgb_blocks[current_stage](x)
      curLevel_x = self.blocks[current_stage](curLevel_x)

      x = F.avg_pool2d(x, kernel_size=2, stride=2)
      prevLevel_x = self.from_rgb_blocks[prev_stage](x)
      x = alpha * curLevel_x + (1 - alpha) * prevLevel_x
      x = self.blocks[prev_stage](x)

    for lev in range(prev_stage + 1, levels):
      x = self.blocks[lev](x)

    return x

if __name__ == '__main__':
  NOISE_DIM = 512
  IMG_CHANNELS = 3
  gen = Generator(z_dim=NOISE_DIM, img_channels=IMG_CHANNELS)
  crit = Critic(img_channels=IMG_CHANNELS)

  for res in [4, 8, 16, 32, 64, 128, 256]:
    step = int(np.log2(res)) - 2
    noise = torch.randn((8, NOISE_DIM, 1, 1))
    z = gen(noise, alpha=0.5, stage=step)
    assert z.shape == (8, IMG_CHANNELS, res, res)
    z = crit(z, alpha=1.0, stage=step)
    assert z.shape == (8, 1, 1, 1)

  summary(gen, (NOISE_DIM, 1, 1), 4, device="cpu")
  summary(crit, (IMG_CHANNELS, 256, 256), 4, device="cpu")