import torch.nn as nn

from gans.utils.helpers import initialize_model
from .building_blocks import downscale_block

class Critic(nn.Module):
  def __init__(self, image_channels, features_disc):
    super(Critic, self).__init__()

    self.critic = nn.Sequential(
      nn.Conv2d(image_channels, features_disc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
      nn.LeakyReLU(0.2),

      # downscale_block(features_disc,      features_disc * 2,   (4, 4), (2, 2), (1, 1)),
      # downscale_block(features_disc * 2,  features_disc * 4,   (4, 4), (2, 2), (1, 1)),
      # downscale_block(features_disc * 4,  features_disc * 8,   (4, 4), (2, 2), (1, 1)),
      # downscale_block(features_disc * 8,  features_disc * 16,  (4, 4), (2, 2), (1, 1)),

      downscale_block(features_disc,      features_disc * 2,   (4, 4), (2, 2), (1, 1)),
      downscale_block(features_disc * 2,  features_disc * 4,   (4, 4), (2, 2), (1, 1)),
      downscale_block(features_disc * 4,  features_disc * 8,   (4, 4), (2, 2), (1, 1)),

      # nn.Conv2d(features_disc * 16, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)) #1x1

      nn.Conv2d(features_disc * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))  # 1x1
    )

    initialize_model(self)

  def forward(self, x):
    return self.critic(x)