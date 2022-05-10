import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import pathlib

import settings
from dataset import PairDataset
from generator_model import Generator
from discriminator_model import Discriminator

from gans.utils.model_saver import load_model, save_model

def train(disc, gen, train_dataloader, opt_discriminator, opt_generator, l1_loss, bce_loss, g_scaler, d_scaler):
  loop = tqdm(train_dataloader, leave=True)

  D_loss = None
  G_loss = None
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

  return D_loss, G_loss


def main():
  disc = Discriminator().to(settings.device)
  gen = Generator().to(settings.device)
  opt_discriminator = optim.Adam(disc.parameters(), lr=settings.LR, betas=(0.5, 0.999))
  opt_generator = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0.5, 0.999))
  bce_loss = nn.BCEWithLogitsLoss()
  l1_loss = nn.L1Loss()

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/gen.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/gen.mod"):
      load_model(f"models/{settings.MODEL_NAME}/gen.mod", gen, opt_generator, settings.LR, settings.device)

    if os.path.exists(f"models/{settings.MODEL_NAME}/disc.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/disc.mod"):
      load_model(f"models/{settings.MODEL_NAME}/disc.mod", disc, opt_discriminator, settings.LR, settings.device)
  except:
    print("Models are incompatible with found model parameters")
    exit(1)

  if settings.GEN_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.GEN_MODEL_WEIGHTS_TO_LOAD, gen, opt_generator, settings.LR, settings.device)
    except:
      print("Generator model weights are incompatible with found model parameters")
      exit(2)

  if settings.DISC_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.DISC_MODEL_WEIGHTS_TO_LOAD, disc, opt_discriminator, settings.LR, settings.device)
    except:
      print("Discriminator model weights are incompatible with found model parameters")
      exit(2)

  train_dataset = PairDataset(root_dir=settings.TRAINING_DATASET_PATH)
  train_dataloader = DataLoader(train_dataset, settings.BATCH_SIZE, True, num_workers=4)
  test_dataset = PairDataset(root_dir=settings.TESTING_DATASET_PATH)
  test_dataloader = DataLoader(test_dataset, 1, False)

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    for epoch in range(settings.START_EPOCH, settings.EPOCHS):
      d_loss, g_loss = train(disc, gen, train_dataloader, opt_discriminator, opt_generator, l1_loss, bce_loss, g_scaler, d_scaler)

      if d_loss is not None and g_loss is not None:
        print(f"Epoch: {epoch}/{settings.EPOCHS} Loss disc: {d_loss:.4f}, Loss gen: {g_loss:.4f}")
        summary_writer.add_scalar("Gen loss", g_loss, global_step=epoch)
        summary_writer.add_scalar("Disc loss", d_loss, global_step=epoch)

      if epoch % 5 == 0:
        save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
        save_model(disc, opt_discriminator, f"models/{settings.MODEL_NAME}/disc.mod")

        save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/{epoch}_gen.mod")
        save_model(disc, opt_discriminator, f"models/{settings.MODEL_NAME}/{epoch}_disc.mod")

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
    save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
    save_model(disc, opt_discriminator, f"models/{settings.MODEL_NAME}/disc.mod")

if __name__ == '__main__':
  main()
