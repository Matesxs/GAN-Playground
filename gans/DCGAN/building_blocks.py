import torch.nn as nn

def upscale_block(in_ch, out_ch, kernel, stride, padding):
  return nn.Sequential(
    nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
    nn.BatchNorm2d(out_ch),
    nn.ReLU()
  )

def downscale_block(in_ch, out_ch, kernel, stride, padding):
  return nn.Sequential(
    nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
    nn.BatchNorm2d(out_ch),
    nn.LeakyReLU(0.2)
  )