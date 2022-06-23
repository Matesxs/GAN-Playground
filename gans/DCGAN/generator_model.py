import torch.nn as nn

from gans.utils.helpers import initialize_model
from .building_blocks import upscale_block

class Generator(nn.Module):
  def __init__(self, noise_dim, image_channels, features_gen):
    super(Generator, self).__init__()

    self.generator = nn.Sequential(
      # upscale_block(noise_dim,         features_gen * 32, (4, 4), (2, 2), (0, 0)),
      # upscale_block(features_gen * 32, features_gen * 16, (4, 4), (2, 2), (1, 1)),
      # upscale_block(features_gen * 16, features_gen * 8,  (4, 4), (2, 2), (1, 1)),
      # upscale_block(features_gen * 8,  features_gen * 4,  (4, 4), (2, 2), (1, 1)),
      # upscale_block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),

      upscale_block(noise_dim, features_gen * 16, (4, 4), (2, 2), (0, 0)),
      upscale_block(features_gen * 16, features_gen * 8,  (4, 4), (2, 2), (1, 1)),
      upscale_block(features_gen * 8,  features_gen * 4,  (4, 4), (2, 2), (1, 1)),
      upscale_block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),

      nn.ConvTranspose2d(features_gen * 2, image_channels, (4, 4), (2, 2), (1, 1)),
      nn.Tanh()
    )

    initialize_model(self)

  def forward(self, x):
    return self.generator(x)