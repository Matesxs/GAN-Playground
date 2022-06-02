import torch
import torch.nn as nn

from helpers import initialize_model
from building_blocks import upscale_block

class Generator(nn.Module):
  def __init__(self, noise_dim, image_channels, features_gen, num_of_classes, img_size, embed_size):
    super(Generator, self).__init__()

    self.img_size = img_size

    self.generator = nn.Sequential(
      upscale_block(noise_dim + embed_size, features_gen * 8, (4, 4), (2, 2), (0, 0)),
      upscale_block(features_gen * 8, features_gen * 4,  (4, 4), (2, 2), (1, 1)),
      upscale_block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),
      upscale_block(features_gen * 2,  features_gen * 1,  (4, 4), (2, 2), (1, 1)),

      # nn.Conv2d(features_gen * 2, features_gen * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
      # nn.InstanceNorm2d(features_gen * 2, affine=True),
      # nn.ReLU(),

      nn.ConvTranspose2d(features_gen * 1, image_channels, (4, 4), (2, 2), (1, 1)),
      nn.Tanh()
    )

    self.embeding = nn.Embedding(num_of_classes, embed_size)

    initialize_model(self)

  def forward(self, x, labels):
    embed = self.embeding(labels).unsqueeze(2).unsqueeze(3)
    x = torch.cat([x, embed], dim=1)
    return self.generator(x)

if __name__ == '__main__':
  x = torch.randn((1, 128, 1, 1))
  labels = torch.randint(9, (1,))

  gen = Generator(128, 3, 64, 10, 64, 128)
  pred = gen(x, labels)
  print(pred.shape)