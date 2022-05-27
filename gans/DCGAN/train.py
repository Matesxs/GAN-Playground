import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import pathlib
from tqdm import tqdm

from generator_model import Generator
from discriminator_model import Discriminator
from settings import *

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def train_step(loader, disc, gen, optimizer_disc, optimizer_gen, loss):
  loop = tqdm(loader, leave=True, unit="batch")

  loss_disc = loss_gen = None
  for batch_idx, (real, _) in enumerate(loop):
    real = real.to(device)
    noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1), device=device)
    fake = gen(noise)

    # Train discriminator
    disc_real = disc(real).reshape(-1)  # Flatten
    loss_disc_real = loss(disc_real, torch.ones_like(disc_real))

    disc_fake = disc(fake).reshape(-1)
    loss_disc_fake = loss(disc_fake, torch.zeros_like(disc_fake))

    loss_disc = (loss_disc_real + loss_disc_fake) / 2
    disc.zero_grad()

    loss_disc.backward(retain_graph=True)
    optimizer_disc.step()

    # Train generator
    output = disc(fake).reshape(-1)
    loss_gen = loss(output, torch.ones_like(output))
    gen.zero_grad()
    loss_gen.backward()
    optimizer_gen.step()

  return loss_disc, loss_gen

def train():
  dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

  gen = Generator(NOISE_DIM, IMG_CH, FEATURES_GEN).to(device)
  disc = Discriminator(IMG_CH, FEATURES_DISC).to(device)

  optimizer_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
  optimizer_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

  try:
    if os.path.exists(f"models/{MODEL_NAME}/gen.mod") and os.path.isfile(f"models/{MODEL_NAME}/gen.mod"):
      load_model(f"models/{MODEL_NAME}/gen.mod", gen, optimizer_gen, LR, device)

    if os.path.exists(f"models/{MODEL_NAME}/disc.mod") and os.path.isfile(f"models/{MODEL_NAME}/disc.mod"):
      load_model(f"models/{MODEL_NAME}/disc.mod", disc, optimizer_disc, LR, device)
  except:
    print("Models are incompatible with found model parameters")

  metadata = None
  if os.path.exists(f"models/{MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{MODEL_NAME}/metadata.pkl")

  start_epoch = 0
  if metadata is not None:
    if "epoch" in metadata.keys():
      start_epoch = int(metadata["epoch"])

  if GEN_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(GEN_MODEL_WEIGHTS_TO_LOAD, gen, optimizer_gen, LR, device)
    except:
      print("Generator model weights are incompatible with found model parameters")
      exit(2)

  if DISC_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(DISC_MODEL_WEIGHTS_TO_LOAD, disc, optimizer_disc, LR, device)
    except:
      print("Discriminator model weights are incompatible with found model parameters")
      exit(2)

  loss = nn.BCELoss()

  test_noise = torch.randn((32, NOISE_DIM, 1, 1), device=device)

  summary_writer_fake = SummaryWriter(f"logs/{MODEL_NAME}")
  summary_writer_values = SummaryWriter(f"logs/{MODEL_NAME}/scalars")

  gen.train()
  disc.train()

  print("Discriminator")
  summary(disc, (IMG_CH, IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

  print("Generator")
  summary(gen, (NOISE_DIM, 1, 1), batch_size=BATCH_SIZE)

  if not os.path.exists(f"models/{MODEL_NAME}"):
    pathlib.Path(f"models/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  last_epoch = 0
  try:
    for epoch in range(start_epoch, EPOCHS):
      last_epoch = epoch

      loss_disc, loss_gen = train_step(loader, disc, gen, optimizer_disc, optimizer_gen, loss)
      if loss_gen is not None and loss_disc is not None:
        print(f"Epoch: {epoch}/{EPOCHS} Loss disc: {loss_disc:.4f}, Loss gen: {loss_gen:.4f}")

        summary_writer_values.add_scalar("Gen Loss", loss_gen, global_step=epoch)
        summary_writer_values.add_scalar("Disc Loss", loss_disc, global_step=epoch)

      if epoch % SAVE_INTERVAL == 0:
        with torch.no_grad():
          fake = gen(test_noise)
          img_grid_fake = torchvision.utils.make_grid(fake[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)
          summary_writer_fake.add_image("Fake", img_grid_fake, global_step=epoch)

        save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen_{epoch}.mod")
        save_model(disc, optimizer_disc, f"models/{MODEL_NAME}/disc_{epoch}.mod")

      save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
      save_model(disc, optimizer_disc, f"models/{MODEL_NAME}/disc.mod")
      save_metadata({"epoch": last_epoch}, f"models/{MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")

  save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
  save_model(disc, optimizer_disc, f"models/{MODEL_NAME}/disc.mod")
  save_metadata({"epoch": last_epoch}, f"models/{MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
    train()