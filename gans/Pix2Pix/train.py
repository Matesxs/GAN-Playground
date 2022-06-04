import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import os
import pathlib

import settings
from generator_model import Generator
from discriminator_model import Discriminator

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata
from gans.utils.datasets import JoinedImagePairDataset

def gradient_penalty(critic, real, fake, device):
  batch, channels, height, width = real.shape
  epsilon = torch.randn((batch, 1, 1, 1)).repeat(1, channels, height, width).to(device)
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

def train(x, y, disc, gen, opt_discriminator, opt_generator, l1_loss, bce_loss, g_scaler, d_scaler):
  x, y = x.to(settings.device), y.to(settings.device)

  # Train discriminator
  with torch.cuda.amp.autocast():
    y_fake = gen(x)

    D_real = disc(x, y)
    D_fake = disc(x, y_fake.detach())

    D_real_loss = bce_loss(D_real, (torch.ones_like(D_real) - 0.1 * torch.rand_like(D_real)) if settings.TRUE_LABEL_SMOOTHING else torch.ones_like(D_real))
    D_fake_loss = bce_loss(D_fake, (torch.zeros_like(D_fake) + 0.1 * torch.rand_like(D_fake)) if settings.FAKE_LABEL_SMOOTHING else torch.zeros_like(D_fake))
    D_loss = D_real_loss + D_fake_loss

    if settings.GP_LAMBDA != 0:
      D_loss += settings.GP_LAMBDA * gradient_penalty(disc, y, y_fake, settings.device)

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
  disc = Discriminator(channels=settings.IMG_CHAN, features=settings.FEATURES_DISC).to(settings.device)
  gen = Generator(in_channels=settings.IMG_CHAN, features=settings.FEATURES_GEN).to(settings.device)
  opt_discriminator = optim.Adam(disc.parameters(), lr=settings.LR, betas=(0.9 if settings.GP_LAMBDA == 0 else 0.5, 0.999 if settings.GP_LAMBDA == 0 else 0.9))
  opt_generator = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0.9 if settings.GP_LAMBDA == 0 else 0.5, 0.999 if settings.GP_LAMBDA == 0 else 0.9))
  bce_loss = nn.BCEWithLogitsLoss()
  l1_loss = nn.L1Loss()

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/gen.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/gen.mod"):
      load_model(f"models/{settings.MODEL_NAME}/gen.mod", gen, opt_generator, settings.LR, settings.device)
  except:
    print("Generator model is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/disc.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/disc.mod"):
      load_model(f"models/{settings.MODEL_NAME}/disc.mod", disc, opt_discriminator, settings.LR, settings.device)
  except:
    print("Discriminator model is incompatible with found model parameters")

  metadata = None
  if os.path.exists(f"models/{settings.MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{settings.MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{settings.MODEL_NAME}/metadata.pkl")

  iteration = 0
  if metadata is not None:
    if "iteration" in metadata.keys():
      iteration = int(metadata["iteration"])

  if settings.GEN_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.GEN_MODEL_WEIGHTS_TO_LOAD, gen, opt_generator, settings.LR, settings.device)
    except Exception as e:
      print(f"Generator model weights are incompatible with found model parameters\n{e}")
      exit(2)

  if settings.DISC_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.DISC_MODEL_WEIGHTS_TO_LOAD, disc, opt_discriminator, settings.LR, settings.device)
    except Exception as e:
      print(f"Discriminator model weights are incompatible with found model parameters\n{e}")
      exit(2)

  dataset_format = "RGB" if settings.IMG_CHAN == 3 else "GRAY"
  train_dataset = JoinedImagePairDataset(root_dir=settings.TRAINING_DATASET_PATH, switch_sides=settings.SWITCH_INPUT_IMAGE_POSITIONS, transform=settings.train_transform, format=dataset_format)
  train_dataloader = DataLoader(train_dataset, settings.BATCH_SIZE, True, num_workers=settings.NUM_OF_WORKERS, persistent_workers=True, pin_memory=True)
  test_dataset = JoinedImagePairDataset(root_dir=settings.TESTING_DATASET_PATH, switch_sides=settings.SWITCH_INPUT_IMAGE_POSITIONS, transform=settings.test_transform, format=dataset_format)
  test_dataloader = DataLoader(test_dataset, settings.TESTING_SAMPLES, False)

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    with tqdm(total=settings.ITERATIONS, initial=iteration, unit="it") as bar:
      while True:
        for x, y in train_dataloader:
          d_loss, g_loss = train(x, y, disc, gen, opt_discriminator, opt_generator, l1_loss, bce_loss, g_scaler, d_scaler)
          if d_loss is not None:
            summary_writer.add_scalar("Disc Loss", d_loss, global_step=iteration)
          if g_loss is not None:
            summary_writer.add_scalar("Gen Loss", g_loss, global_step=iteration)

          if settings.SAVE_CHECKPOINT and iteration % settings.CHECKPOINT_EVERY == 0:
            save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/{iteration}_gen.mod")
            save_model(disc, opt_discriminator, f"models/{settings.MODEL_NAME}/{iteration}_disc.mod")

          if iteration % settings.SAMPLE_EVERY == 0:
            print(f"Loss disc: {d_loss:.4f}, Loss gen: {g_loss:.4f}")

            x, y = next(iter(test_dataloader))
            x, y = x.to(settings.device), y.to(settings.device)
            gen.eval()

            with torch.no_grad():
              y_fake = gen(x)
              img_grid_fake = torchvision.utils.make_grid(y_fake[:settings.TESTING_SAMPLES], normalize=True)
              summary_writer.add_image("Generated", img_grid_fake, global_step=iteration)

              if iteration == 0:
                img_grid_input = torchvision.utils.make_grid(x[:settings.TESTING_SAMPLES], normalize=True)
                img_grid_real = torchvision.utils.make_grid(y[:settings.TESTING_SAMPLES], normalize=True)
                summary_writer.add_image("Real", img_grid_real, global_step=iteration)
                summary_writer.add_image("Input", img_grid_input, global_step=iteration)

            gen.train()

          bar.update()
          iteration += 1
          if iteration > settings.ITERATIONS:
            break

        if iteration > settings.ITERATIONS:
          break

        save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
        save_model(disc, opt_discriminator, f"models/{settings.MODEL_NAME}/disc.mod")
        save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")
  except Exception as e:
    print(e)
    save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
    exit(-1)

  save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
  save_model(disc, opt_discriminator, f"models/{settings.MODEL_NAME}/disc.mod")
  save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
  main()
