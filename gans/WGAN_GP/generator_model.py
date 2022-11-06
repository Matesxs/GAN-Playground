import torch.nn as nn

from gans.utils.helpers import initialize_model
from gans.utils.global_modules import PixelShuffleConv
from torchsummary import summary

def upscale_block(in_ch, out_ch, kernel, stride, padding):
  return nn.Sequential(
    nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
    nn.BatchNorm2d(out_ch),
    nn.ReLU()
  )

def upscale_block_pix_shuffle(in_ch, out_ch, kernel, scale, padding):
  return nn.Sequential(
    PixelShuffleConv(in_ch, scale, kernel, padding, out_ch),
    nn.BatchNorm2d(out_ch),
    nn.ReLU()
  )

class Generator(nn.Module):
  def __init__(self, noise_dim, image_channels, features, scale_map, pixel_shuffle=False):
    super(Generator, self).__init__()

    # self.generator = nn.Sequential(
    #   # upscale_block(noise_dim,         features_gen * 16, (4, 4), (2, 2), (0, 0)),
    #   # upscale_block(features_gen * 16, features_gen * 8, (4, 4), (2, 2), (1, 1)),
    #   # upscale_block(features_gen * 8, features_gen * 4,  (4, 4), (2, 2), (1, 1)),
    #   # upscale_block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),
    #   # upscale_block(features_gen * 2,  features_gen * 1,  (4, 4), (2, 2), (1, 1)),
    #
    #   upscale_block(noise_dim, features_gen * 8, (4, 4), (2, 2), (0, 0)),
    #   upscale_block(features_gen * 8, features_gen * 4,  (4, 4), (2, 2), (1, 1)),
    #   upscale_block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),
    #   upscale_block(features_gen * 2,  features_gen * 1,  (4, 4), (2, 2), (1, 1)),
    #
    #   nn.ConvTranspose2d(features_gen * 1, image_channels, (4, 4), (2, 2), (1, 1)),
    #   nn.Tanh()
    # )

    layers = []
    if pixel_shuffle:
      for i, (out_channels, scale) in enumerate(zip(features, scale_map)):
        if i == 0:
          layers += [nn.Unflatten(1, (noise_dim, 1, 1)),
                     upscale_block_pix_shuffle(noise_dim, out_channels, kernel=3, scale=scale, padding=1)]
        else:
          in_channels = features[i - 1]
          layers.append(upscale_block_pix_shuffle(in_channels, out_channels, kernel=3, scale=scale, padding=1))

      layers += [
        nn.Conv2d(features[-1], image_channels * 2 ** 2, (3, 3), 1, (1, 1)),
        nn.PixelShuffle(2),
        nn.Tanh()
      ]
    else:
      for i, (out_channels, scale) in enumerate(zip(features, scale_map)):
        if i == 0:
          layers += [nn.Unflatten(1, (noise_dim, 1, 1)),
                     upscale_block(noise_dim, out_channels, 4, scale, 0)]
        else:
          in_channels = features[i - 1]
          layers.append(upscale_block(in_channels, out_channels, 4, scale, 1))

      layers += [
        nn.Conv2d(features[-1], image_channels, (3, 3), 1, (1, 1)),
        nn.Tanh()
      ]

    self.generator = nn.Sequential(*layers)

    initialize_model(self)

  def forward(self, x):
    return self.generator(x)

if __name__ == '__main__':
  from gans.WGAN_GP import settings

  gen = Generator(settings.NOISE_DIM, settings.IMG_CH, settings.FEATURES_GEN, settings.SCALE_OR_STRIDE_FACTOR_GEN, settings.GENERATOR_USE_PIXELSHUFFLE)

  summary(gen, (settings.NOISE_DIM,), batch_size=settings.BATCH_SIZE, device="cpu")