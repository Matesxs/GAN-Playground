import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device):
  batch, channels, height, width = real.shape
  epsilon = torch.randn((batch, 1, 1, 1)).repeat(1, channels, height, width).to(device)
  interpolated_imgs = real * epsilon + fake * (1 - epsilon)

  interpolated_scores = critic(interpolated_imgs)
  gradient = torch.autograd.grad(inputs=interpolated_imgs, outputs=interpolated_scores,
                                 grad_outputs=torch.ones_like(interpolated_scores),
                                 create_graph=True, retain_graph=True)[0]

  # Flattening of gradient
  gradient = gradient.view(gradient.shape[0], -1)
  gradient_norm = gradient.norm(2, dim=1) # L2 norm

  return torch.mean((gradient_norm - 1) ** 2)

class PixelNorm(nn.Module):
  def __init__(self, epsilon=1e-8):
    super(PixelNorm, self).__init__()

    self.epsilon = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
