import torch
import torch.nn as nn

from gans.utils.helpers import initialize_model

def downscale_block(in_ch, out_ch, kernel, stride, padding):
  return nn.Sequential(
    nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
    nn.InstanceNorm2d(out_ch, affine=True),
    nn.LeakyReLU(0.2)
  )

class Critic(nn.Module):
  def __init__(self, image_channels, features_disc, num_of_classes, img_size):
    super(Critic, self).__init__()

    self.img_size = img_size

    self.critic = nn.Sequential(
      nn.Conv2d(image_channels + 1, features_disc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
      nn.LeakyReLU(0.2),

      downscale_block(features_disc,      features_disc * 2,   (4, 4), (2, 2), (1, 1)),
      downscale_block(features_disc * 2,  features_disc * 4,   (4, 4), (2, 2), (1, 1)),
      downscale_block(features_disc * 4,  features_disc * 8,   (4, 4), (2, 2), (1, 1)),

      # nn.Conv2d(features_disc * 8, features_disc * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
      # nn.InstanceNorm2d(features_disc * 8, affine=True),
      # nn.LeakyReLU(0.2),

      nn.Conv2d(features_disc * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
    )

    self.embeding = nn.Embedding(num_of_classes, img_size * img_size)

    initialize_model(self)

  def forward(self, x, labels):
    embed = self.embeding(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
    x = torch.cat([x, embed], dim=1)
    return self.critic(x)

if __name__ == '__main__':
  x = torch.randn((1, 3, 64, 64))
  labels = torch.randint(9, (1,))

  critic = Critic(3, 64, 10, 64)
  pred = critic(x, labels)
  print(pred.shape)
