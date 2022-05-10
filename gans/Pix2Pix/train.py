import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

import settings
from dataset import Pix2PixDataset
from generator_model import Generator
from discriminator_model import Discriminator

def train(disc, gen, train_dataloader, opt_discriminator, opt_generator, l1_loss, bce_loss, g_scaler, d_scaler):
  loop = tqdm(train_dataloader, leave=True)

  for idx, (x, y) in enumerate(loop):
    x, y = x.to(settings.device), y.to(settings.device)

    # Train discriminator
    with torch.cuda.amp.autocast():
      y_fake = gen(x)
      D_real = disc(x, y)
      D_fake = disc(x, y_fake.detach())

      D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
      D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
      D_loss = (D_real_loss + D_fake_loss) / 2

    disc.zero_grad()
    d_scaler.scale(D_loss).backward()
    d_scaler.step(opt_discriminator)
    d_scaler.update()

    # Train generator
    with torch.cuda.amp.autocast():
      D_fake = disc(x, y_fake)
      G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
      L1 = l1_loss(y_fake, y) * settings.L1_LAMBDA
      G_loss = G_fake_loss + L1

    opt_generator.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(opt_generator)
    g_scaler.update()


def main():
  disc = Discriminator().to(settings.device)
  gen = Generator().to(settings.device)
  opt_discriminator = optim.Adam(disc.parameters(), lr=settings.LR, betas=(0.5, 0.999))
  opt_generator = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0.5, 0.999))
  bce_loss = nn.BCEWithLogitsLoss()
  l1_loss = nn.L1Loss()

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}_gen.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}_gen.mod"):
      gen.load_state_dict(torch.load(f"models/{settings.MODEL_NAME}_gen.mod"))

    if os.path.exists(f"models/{settings.MODEL_NAME}_disc.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}_disc.mod"):
      disc.load_state_dict(torch.load(f"models/{settings.MODEL_NAME}_disc.mod"))
  except:
    print("Models are incompatible with found model parameters")
    exit(1)

  train_dataset = Pix2PixDataset(root_dir="datasets/maps/train")
  train_dataloader = DataLoader(train_dataset, settings.BATCH_SIZE, True, num_workers=4)
  test_dataset = Pix2PixDataset(root_dir="datasets/maps/val")
  test_dataloader = DataLoader(test_dataset, 1, False)

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  try:
    for epoch in range(settings.START_EPOCH, settings.EPOCHS):
      train(disc, gen, train_dataloader, opt_discriminator, opt_generator, l1_loss, bce_loss, g_scaler, d_scaler)

      if epoch % 5 == 0:
        torch.save(gen.state_dict(), f"models/{settings.MODEL_NAME}_gen.mod")
        torch.save(disc.state_dict(), f"models/{settings.MODEL_NAME}_disc.mod")

      x, y = next(iter(test_dataloader))
      x, y = x.to(settings.device), y.to(settings.device)
      gen.eval()

      with torch.no_grad():
        y_fake = gen(x)

        summary_writer.add_image("Input", x[0] * 0.5 + 0.5, global_step=epoch)
        summary_writer.add_image("Generated", y_fake[0] * 0.5 + 0.5, global_step=epoch)
        summary_writer.add_image("Real", y[0] * 0.5 + 0.5, global_step=epoch)

      gen.train()
  except KeyboardInterrupt:
    pass
  finally:
    torch.save(gen.state_dict(), f"models/{settings.MODEL_NAME}_gen.mod")
    torch.save(disc.state_dict(), f"models/{settings.MODEL_NAME}_disc.mod")

if __name__ == '__main__':
  main()
