import torch.nn as nn
from gans.utils.global_modules import PixelShuffleConv

def downscale_block(in_ch, out_ch, kernel, stride, padding):
  return nn.Sequential(
    nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
    nn.InstanceNorm2d(out_ch, affine=True),
    nn.LeakyReLU(0.2)
  )

def upscale_block(in_ch, out_ch, kernel, stride, padding):
  return nn.Sequential(
    nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
    nn.BatchNorm2d(out_ch),
    nn.ReLU()
  )

def upscale_block_pix_shuffle(in_ch, out_ch, kernel, stride, padding):
  return nn.Sequential(
    PixelShuffleConv(in_ch, stride if isinstance(stride, int) else stride[0], kernel, padding, out_ch),
    nn.BatchNorm2d(out_ch),
    nn.ReLU()
  )