import torch
import torch.nn as nn
from torchsummary import summary

def initialize_model(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

class Critic(nn.Module):
  def __init__(self, image_channels, features_disc, num_of_classes, img_size):
    super(Critic, self).__init__()

    self.img_size = img_size

    self.critic = nn.Sequential(
      nn.Conv2d(image_channels + 1, features_disc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
      nn.LeakyReLU(0.2),

      self._block(features_disc,      features_disc * 2,   (4, 4), (2, 2), (1, 1)),
      self._block(features_disc * 2,  features_disc * 4,   (4, 4), (2, 2), (1, 1)),
      self._block(features_disc * 4,  features_disc * 8,   (4, 4), (2, 2), (1, 1)),

      nn.Conv2d(features_disc * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
    )

    self.embeding = nn.Embedding(num_of_classes, img_size * img_size)

    initialize_model(self)

  @staticmethod
  def _block(in_ch, out_ch, kernel, stride, padding):
    return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
      nn.InstanceNorm2d(out_ch, affine=True),
      nn.LeakyReLU(0.2)
    )

  def forward(self, x, labels):
    embed = self.embeding(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
    x = torch.cat([x, embed], dim=1)
    return self.critic(x)


class Generator(nn.Module):
  def __init__(self, noise_dim, image_channels, features_gen, num_of_classes, img_size, embed_size):
    super(Generator, self).__init__()

    self.img_size = img_size

    self.generator = nn.Sequential(
      self._block(noise_dim + embed_size, features_gen * 16, (4, 4), (2, 2), (0, 0)),
      self._block(features_gen * 16, features_gen * 8,  (4, 4), (2, 2), (1, 1)),
      self._block(features_gen * 8,  features_gen * 4,  (4, 4), (2, 2), (1, 1)),
      self._block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),

      nn.ConvTranspose2d(features_gen * 2, image_channels, (4, 4), (2, 2), (1, 1)),
      nn.Tanh()
    )

    self.embeding = nn.Embedding(num_of_classes, embed_size)

    initialize_model(self)

  @staticmethod
  def _block(in_ch, out_ch, kernel, stride, padding):
    return nn.Sequential(
      nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU()
    )

  def forward(self, x, labels):
    embed = self.embeding(labels).unsqueeze(2).unsqueeze(3)
    x = torch.cat([x, embed], dim=1)
    return self.generator(x)
