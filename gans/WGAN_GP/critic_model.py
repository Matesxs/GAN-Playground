import torch.nn as nn

from gans.utils.helpers import initialize_model
from torchsummary import summary

def downscale_block(in_ch, out_ch, kernel, stride, padding, use_norm=True):
  return nn.Sequential(
    nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False, padding_mode="reflect"),
    nn.InstanceNorm2d(out_ch, affine=True) if use_norm else nn.Identity(),
    nn.LeakyReLU(0.2)
  )

class Critic(nn.Module):
  def __init__(self, image_channels, features, strides_map):
    super(Critic, self).__init__()

    layers = []
    for i, (out_channels, strides) in enumerate(zip(features, strides_map)):
      if i == 0:
        layers.append(downscale_block(image_channels, out_channels, kernel=4, stride=strides, padding=1, use_norm=False))
      else:
        in_channels = features[i - 1]
        layers.append(downscale_block(in_channels, out_channels, kernel=4, stride=strides, padding=1))

    layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=0, padding_mode="reflect"))

    self.critic = nn.Sequential(*layers)

    initialize_model(self)

  def forward(self, x):
    return self.critic(x)

if __name__ == '__main__':
  from gans.WGAN_GP import settings

  crit = Critic(settings.IMG_CH, settings.FEATURES_CRIT, settings.STRIDES_CRIT)

  summary(crit, (settings.IMG_CH, settings.IMG_SIZE, settings.IMG_SIZE), batch_size=settings.BATCH_SIZE, device="cpu")