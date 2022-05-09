import torch
import torch.nn as nn

from helpers import initialize_model
from building_blocks import upscale_block

class Generator(nn.Module):
  def __init__(self, noise_dim, image_channels, features_gen, num_of_classes, img_size, embed_size):
    super(Generator, self).__init__()

    self.img_size = img_size

    self.generator = nn.Sequential(
      upscale_block(noise_dim + embed_size, features_gen * 16, (4, 4), (2, 2), (0, 0)),
      upscale_block(features_gen * 16, features_gen * 8,  (4, 4), (2, 2), (1, 1)),
      upscale_block(features_gen * 8,  features_gen * 4,  (4, 4), (2, 2), (1, 1)),
      upscale_block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),

      nn.ConvTranspose2d(features_gen * 2, image_channels, (4, 4), (2, 2), (1, 1)),
      nn.Tanh()
    )

    self.embeding = nn.Embedding(num_of_classes, embed_size)

    initialize_model(self)

  def forward(self, x, labels):
    embed = self.embeding(labels).unsqueeze(2).unsqueeze(3)
    x = torch.cat([x, embed], dim=1)
    return self.generator(x)
