import torch
import torch.nn as nn
from torchsummary import summary

def initialize_model(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

class Discriminator(nn.Module):
  def __init__(self, image_channels, features_disc):
    super(Discriminator, self).__init__()

    self.discriminator = nn.Sequential(
      nn.Conv2d(image_channels, features_disc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
      nn.LeakyReLU(0.2),

      self._block(features_disc,      features_disc * 2,   (4, 4), (2, 2), (1, 1)),
      self._block(features_disc * 2,  features_disc * 4,   (4, 4), (2, 2), (1, 1)),
      self._block(features_disc * 4,  features_disc * 8,   (4, 4), (2, 2), (1, 1)),
      self._block(features_disc * 8,  features_disc * 16,  (4, 4), (2, 2), (1, 1)),

      # self._block(features_disc,      features_disc * 2,   (4, 4), (2, 2), (1, 1)),
      # self._block(features_disc * 2,  features_disc * 4,   (4, 4), (2, 2), (1, 1)),
      # self._block(features_disc * 4,  features_disc * 8,   (4, 4), (2, 2), (1, 1)),

      nn.Conv2d(features_disc * 16, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)), #1x1
      nn.Sigmoid()

      # nn.Conv2d(features_disc * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # 1x1
      # nn.Sigmoid()
    )

    initialize_model(self)

  @staticmethod
  def _block(in_ch, out_ch, kernel, stride, padding):
    return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.discriminator(x)


class Generator(nn.Module):
  def __init__(self, noise_dim, image_channels, features_gen):
    super(Generator, self).__init__()

    self.generator = nn.Sequential(
      self._block(noise_dim,         features_gen * 32, (4, 4), (2, 2), (0, 0)),
      self._block(features_gen * 32, features_gen * 16, (4, 4), (2, 2), (1, 1)),
      self._block(features_gen * 16, features_gen * 8,  (4, 4), (2, 2), (1, 1)),
      self._block(features_gen * 8,  features_gen * 4,  (4, 4), (2, 2), (1, 1)),
      self._block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),

      # self._block(noise_dim, features_gen * 16, (4, 4), (2, 2), (0, 0)),
      # self._block(features_gen * 16, features_gen * 8,  (4, 4), (2, 2), (1, 1)),
      # self._block(features_gen * 8,  features_gen * 4,  (4, 4), (2, 2), (1, 1)),
      # self._block(features_gen * 4,  features_gen * 2,  (4, 4), (2, 2), (1, 1)),

      nn.ConvTranspose2d(features_gen * 2, image_channels, (4, 4), (2, 2), (1, 1)),
      nn.Tanh()
    )

    initialize_model(self)

  @staticmethod
  def _block(in_ch, out_ch, kernel, stride, padding):
    return nn.Sequential(
      nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU()
    )

  def forward(self, x):
    return self.generator(x)


if __name__ == '__main__':
  batch = 8
  in_ch = 3
  img_size = 64
  noise_dim = 128

  x = torch.randn((batch, in_ch, img_size, img_size), device="cuda")
  y = torch.randn((batch, noise_dim, 1, 1), device="cuda")

  disc = Discriminator(in_ch, 64).to("cuda")
  gen = Generator(noise_dim, in_ch, 64).to("cuda")

  print(f"Discriminator shape: {disc(x).shape}")
  summary(disc, (in_ch, img_size, img_size), batch)

  print(f"Generator shape: {gen(y).shape}")
  summary(gen, (noise_dim, 1, 1), batch)