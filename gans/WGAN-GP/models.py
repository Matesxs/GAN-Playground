import torch
import torch.nn as nn
from torchsummary import summary









if __name__ == '__main__':
  batch = 8
  in_ch = 3
  img_size = 64
  noise_dim = 128

  x = torch.randn((batch, in_ch, img_size, img_size), device="cuda")
  y = torch.randn((batch, noise_dim, 1, 1), device="cuda")

  disc = Critic(in_ch, 64).to("cuda")
  gen = Generator(noise_dim, in_ch, 64).to("cuda")

  print(f"Critic shape: {disc(x).shape}")
  summary(disc, (in_ch, img_size, img_size), batch)

  print(f"Generator shape: {gen(y).shape}")
  summary(gen, (noise_dim, 1, 1), batch)