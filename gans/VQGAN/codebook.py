import torch
import torch.nn as nn

class Codebook(nn.Module):
  def __init__(self, num_of_vectors: int, latent_dimension: int, beta: float):
    super(Codebook, self).__init__()

    self.beta = beta
    self.latent_dimension = latent_dimension

    self.embeding = nn.Embedding(num_of_vectors, latent_dimension)
    self.embeding.weight.data.uniform_(-1.0 / num_of_vectors, 1.0 / num_of_vectors)

  def forward(self, z):
    z = z.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.latent_dimension)

    d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
        torch.sum(self.embeding.weight**2, dim=1) - \
        2 * (torch.matmul(z_flattened, self.embeding.weight.t()))

    min_encoding_indices = torch.argmin(d, dim=1)
    z_q = self.embeding(min_encoding_indices).view(z.shape)

    loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

    z_q = z + (z_q - z).detach()

    z_q = z_q.permute(0, 3, 1, 2)

    return z_q, min_encoding_indices, loss
