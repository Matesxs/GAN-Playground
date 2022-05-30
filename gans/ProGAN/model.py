import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

from settings import IMG_SIZE
from gans.utils.global_modules import PixelNorm

class WeightedScaleConv2(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2, use_conv=True):
    super(WeightedScaleConv2, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) if use_conv else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
    self.bias = self.conv.bias
    self.conv.bias = None

    nn.init.normal_(self.conv.weight)
    nn.init.zeros_(self.bias)

  def forward(self, x):
    return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, use_pixelnorm=True):
    super(ConvBlock, self).__init__()

    self.block = nn.Sequential(
      WeightedScaleConv2(in_channels, out_channels),
      nn.LeakyReLU(0.2),
      PixelNorm() if use_pixelnorm else nn.Identity(),
      WeightedScaleConv2(out_channels, out_channels),
      nn.LeakyReLU(0.2),
      PixelNorm() if use_pixelnorm else nn.Identity()
    )

  def forward(self, x):
    return self.block(x)

class Generator(nn.Module):
  def __init__(self, z_dim, channels, img_channels=3):
    super(Generator, self).__init__()

    self.initial = nn.Sequential(
      PixelNorm(),
      nn.ConvTranspose2d(z_dim, channels[0], 4, 1, 0),
      # WeightedScaleConv2(z_dim, channels[0], 4, 1, 0, use_conv=False),
      nn.LeakyReLU(0.2),
      WeightedScaleConv2(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.2),
      PixelNorm()
    )

    self.initial_rgb = WeightedScaleConv2(channels[0], img_channels, kernel_size=1, stride=1, padding=0)

    self.blocks = nn.ModuleList()
    self.rgb_layers = nn.ModuleList([self.initial_rgb])

    for i in range(len(channels) - 1):
      in_ch = channels[i]
      out_ch = channels[i + 1]
      self.blocks.append(ConvBlock(in_ch, out_ch))
      self.rgb_layers.append(WeightedScaleConv2(out_ch, img_channels, kernel_size=1, stride=1, padding=0))

  def fade_in(self, alpha, upscaled, generated):
    return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

  def forward(self, x, alpha=1.0, steps=None):
    if steps is None:
      steps = int(log2(IMG_SIZE / 4))

    out = self.initial(x)

    if steps == 0:
      return self.initial_rgb(out)

    upscale = F.interpolate(out, scale_factor=2, mode="nearest")
    out = self.blocks[0](upscale)

    for step in range(1, steps):
      upscale = F.interpolate(out, scale_factor=2, mode="nearest")
      out = self.blocks[step](upscale)

    final_upscaled = self.rgb_layers[steps - 1](upscale)
    final_out = self.rgb_layers[steps](out)
    return self.fade_in(alpha, final_upscaled, final_out)

class Critic(nn.Module):
  def __init__(self, channels, img_channels=3):
    super(Critic, self).__init__()

    self.blocks = nn.ModuleList()
    self.rgb_layers = nn.ModuleList()

    channels = list(reversed(channels))

    for i in range(len(channels) - 1):
      in_ch = channels[i]
      out_ch = channels[i + 1]
      self.blocks.append(ConvBlock(in_ch, out_ch, use_pixelnorm=False))
      self.rgb_layers.append(WeightedScaleConv2(img_channels, in_ch, kernel_size=1, stride=1, padding=0))

    self.initial_rgb = WeightedScaleConv2(img_channels, channels[-1], kernel_size=1, stride=1, padding=0)
    self.rgb_layers.append(self.initial_rgb)
    self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.leaky = nn.LeakyReLU(0.2)

    self.final_block = nn.Sequential(
      WeightedScaleConv2(channels[-1] + 1, channels[-1], kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.2),
      WeightedScaleConv2(channels[-1], channels[-1], kernel_size=4, padding=0, stride=1),
      nn.LeakyReLU(0.2),
      WeightedScaleConv2(channels[-1], 1, kernel_size=1, padding=0, stride=1)
    )

  def fade_in(self, alpha, downscaled, out):
    return alpha * out + (1 - alpha) * downscaled

  def minibatch_std(self, x):
    batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.cat([x, batch_statistics], dim=1)

  def forward(self, x, alpha, steps):
    cur_step = len(self.blocks) - steps
    out = self.leaky(self.rgb_layers[cur_step](x))

    if steps == 0:
      out = self.minibatch_std(out)
      return self.final_block(out).view(out.shape[0], -1)

    downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
    out = self.avg_pool(self.blocks[cur_step](out))
    out = self.fade_in(alpha, downscaled, out)

    for step in range(cur_step + 1, len(self.blocks)):
      out = self.blocks[step](out)
      out = self.avg_pool(out)

    out = self.minibatch_std(out)
    return self.final_block(out).view(out.shape[0], -1)

if __name__ == '__main__':
  Z_DIM = 128
  CHANNELS = [1024, 512, 256, 128, 64, 32]

  gen = Generator(Z_DIM, CHANNELS)
  critic = Critic(CHANNELS)

  for img_size in [4, 8, 16, 32, 64, 128]:
    num_steps = int(log2(img_size / 4))
    x = torch.randn((1, Z_DIM, 1, 1))
    z = gen(x, 0.5, steps=num_steps)
    assert z.shape == (1, 3, img_size, img_size)
    out = critic(z, alpha=0.5, steps=num_steps)
    assert out.shape == (1, 1)

  # input_names = ['Sentence']
  # output_names = ['yhat']
  # torch.onnx.export(gen, (x, 1.0, len(CHANNELS) - 1), 'gen.onnx', input_names=input_names, output_names=output_names)
  # torch.onnx.export(critic, (z, 1.0, len(CHANNELS) - 1), 'crit.onnx', input_names=input_names, output_names=output_names)
