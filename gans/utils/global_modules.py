import torch
import torch.nn as nn

class PixelNorm(nn.Module):
  def __init__(self, epsilon=1e-8):
    super(PixelNorm, self).__init__()

    self.epsilon = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
