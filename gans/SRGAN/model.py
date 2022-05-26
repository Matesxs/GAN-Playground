import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, leaky=False, use_activation=True, use_backnorm=True, **kwargs):
    super(ConvBlock, self).__init__()

    self.block = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_backnorm),
      nn.InstanceNorm2d(out_channels, affine=True) if use_backnorm else nn.Identity(),
      (nn.LeakyReLU(0.2, inplace=True) if leaky else nn.PReLU(num_parameters=out_channels)) if use_activation else nn.Identity()
    )

  def forward(self, x):
    return self.block(x)

class UpsampleBlock(nn.Module):
  def __init__(self, in_channels, scale_factor=2):
    super(UpsampleBlock, self).__init__()

    self.block = nn.Sequential(
      nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1),
      nn.PixelShuffle(scale_factor),
      nn.PReLU(num_parameters=in_channels)
    )

  def forward(self, x):
    return self.block(x)

class ResidualBlock(nn.Module):
  def __init__(self, in_channels):
    super(ResidualBlock, self).__init__()

    self.block = nn.Sequential(
      ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
      ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_activation=False)
    )

  def forward(self, x):
    # Residual block with skip connection
    return self.block(x) + x

class Generator(nn.Module):
  def __init__(self, in_channels=3, features=64, num_blocks=16):
    super(Generator, self).__init__()

    self.initial = ConvBlock(in_channels, features, kernel_size=9, stride=1, padding=4, use_backnorm=False)
    self.residuals = nn.Sequential(*[ResidualBlock(features) for _ in range(num_blocks)])
    self.prep_block = ConvBlock(features, features, kernel_size=3, stride=1, padding=1, use_activation=False)
    self.upscales = nn.Sequential(
      UpsampleBlock(features, scale_factor=2),
      UpsampleBlock(features, scale_factor=2)
    )
    self.final = nn.Sequential(
      nn.Conv2d(features, in_channels, kernel_size=9, stride=1, padding=4),
      nn.Tanh()
    )

  def forward(self, x):
    initial = self.initial(x)
    x = self.residuals(initial)
    x = self.prep_block(x) + initial
    x = self.upscales(x)
    return self.final(x)

class Discriminator(nn.Module):
  def __init__(self, in_channels=3, features=None):
    super(Discriminator, self).__init__()
    
    if features is None:
      features = [64, 64, 128, 128, 256, 256, 512, 512]

    blocks = []
    for idx, feature in enumerate(features):
      blocks.append(ConvBlock(in_channels, feature, kernel_size=3, stride=1 + idx % 2, padding=1, use_activation=True, leaky=True, use_backnorm=False if idx == 0 else True))
      in_channels = feature

    self.blocks = nn.Sequential(*blocks)
    self.classifier = nn.Sequential(
      nn.AdaptiveAvgPool2d((6, 6)),
      nn.Flatten(),
      nn.Linear(512*6*6, 1024),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(1024, 1)
    )

  def forward(self, x):
    x = self.blocks(x)
    return self.classifier(x)

if __name__ == '__main__':
  x = torch.randn((1, 3, 256, 256))

  gen = Generator()
  disc = Discriminator()

  pred = gen(x)
  assert pred.shape == (1, 3, 1024, 1024)

  pred = disc(pred)
  assert pred.shape == (1, 1)

  # input_names = ['Sentence']
  # output_names = ['yhat']
  # torch.onnx.export(gen, x, 'gen.onnx', input_names=input_names, output_names=output_names)
  # torch.onnx.export(disc, x, 'disc.onnx', input_names=input_names, output_names=output_names)
