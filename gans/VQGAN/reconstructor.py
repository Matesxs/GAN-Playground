import torch
import torch.nn as nn
from typing import List, Any

from gans.VQGAN.encoder import Encoder
from gans.VQGAN.decoder import Decoder
from gans.VQGAN.codebook import Codebook

class Reconstructor(nn.Module):
  def __init__(self, encoder_filters: List[int], decoder_filters: List[int], attention_resolutions: List[int], encoder_residuals_per_level: int, decoder_residuals_per_level: int, image_size: int, image_channels: int, latent_dimension: int, latent_vectors: int, beta: float, device: Any):
    super(Reconstructor, self).__init__()

    self.encoder = Encoder(encoder_filters, attention_resolutions, encoder_residuals_per_level, image_size, image_channels, latent_dimension).to(device)
    self.decoder = Decoder(decoder_filters, attention_resolutions, decoder_residuals_per_level, self.encoder.get_final_resolution(), image_channels, latent_dimension).to(device)
    self.codebook = Codebook(latent_vectors, latent_dimension, beta).to(device)
    self.quant_conv = nn.Conv2d(latent_dimension, latent_dimension, kernel_size=1).to(device)
    self.post_quant_conv = nn.Conv2d(latent_dimension, latent_dimension, kernel_size=1).to(device)

  def encode(self, imgs):
    encoded_images = self.encoder(imgs)
    quant_conv_encoded_images = self.quant_conv(encoded_images)
    return self.codebook(quant_conv_encoded_images)

  def decode(self, z):
    post_quant_conv_mapping = self.post_quant_conv(z)
    return self.decoder(post_quant_conv_mapping)

  def forward(self, imgs):
    encoded_images = self.encoder(imgs)
    quant_conv_encoded_images = self.quant_conv(encoded_images)
    codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
    post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
    decoded_images = self.decoder(post_quant_conv_mapping)

    return decoded_images, codebook_indices, q_loss

  def calculate_lambda(self, perceptual_loss, gan_loss):
    last_layer = self.decoder.model[-1]
    last_layer_weight = last_layer.weight
    perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
    gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

    l = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
    l = torch.clamp(l, 0, 1e4).detach()
    return l * 0.8
