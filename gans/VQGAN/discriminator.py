import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, base_filters: int, max_filter_multiplier: int, num_of_layers: int, image_channels: int):
    super(Discriminator, self).__init__()

    layers = [nn.Conv2d(image_channels, base_filters, kernel_size=4, stride=2, padding=1),
              nn.LeakyReLU(0.2)]

    filters_multiplier = 1
    for i in range(1, num_of_layers + 1):
      last_filters_multiplier = filters_multiplier
      filters_multiplier = min(2**i, max_filter_multiplier)

      layers.append(nn.Conv2d(base_filters * last_filters_multiplier, base_filters * filters_multiplier, kernel_size=4, stride=2 if i < num_of_layers else 1, padding=1, bias=False))
      layers.append(nn.BatchNorm2d(base_filters * filters_multiplier)),
      layers.append(nn.LeakyReLU(0.2, True))

    layers.append(nn.Conv2d(base_filters * filters_multiplier, 1, kernel_size=4, stride=1, padding=1))

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)
