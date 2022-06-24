import torch
import torch.nn as nn
from torchvision.models import vgg19
from . import settings

class VGGLoss(nn.Module):
  def __init__(self):
    super(VGGLoss, self).__init__()

    self.vgg = vgg19(pretrained=True).features[:36].eval().to(settings.device)
    self.loss = nn.MSELoss()

    for param in self.vgg.parameters():
      param.requires_grad = False

  def forward(self, upscaled, target):
    vgg_upscaled_features = self.vgg(upscaled)
    vgg_target_features = self.vgg(target)
    return self.loss(vgg_upscaled_features, vgg_target_features)

def gradient_penalty(critic, real, fake, device):
  batch, channels, height, width = real.shape
  epsilon = torch.rand((batch, 1, 1, 1)).repeat(1, channels, height, width).to(device)
  interpolated_images = real * epsilon + fake.detach() * (1 - epsilon)
  interpolated_images.requires_grad_(True)

  interpolated_scores = critic(interpolated_images)

  gradient = torch.autograd.grad(
    inputs=interpolated_images,
    outputs=interpolated_scores,
    grad_outputs=torch.ones_like(interpolated_scores),
    create_graph=True,
    retain_graph=True,
  )[0]

  # Flattening of gradient
  gradient = gradient.view(gradient.shape[0], -1)
  gradient_norm = gradient.norm(2, dim=1)
  gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

  return gradient_penalty
