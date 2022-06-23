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

from gans.DCGAN.generator_model import Generator
from gans.DCGAN.discriminator_model import Discriminator
from gans.DCGAN.settings import *

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def train_step(data, disc, gen, optimizer_disc, optimizer_gen, loss, d_scaler, g_scaler):
  real = data.to(device)
  noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1), device=device)

  # Train discriminator
  with torch.cuda.amp.autocast():
    fake = gen(noise)
    disc_real = disc(real).reshape(-1)  # Flatten
    loss_disc_real = loss(disc_real, torch.ones_like(disc_real))

    disc_fake = disc(fake.detach()).reshape(-1)
    loss_disc_fake = loss(disc_fake, torch.zeros_like(disc_fake))

    loss_disc = (loss_disc_real + loss_disc_fake) / 2

  optimizer_disc.zero_grad()
  d_scaler.scale(loss_disc).backward()
  d_scaler.step(optimizer_disc)
  d_scaler.update()

  # Train generator
  with torch.cuda.amp.autocast():
    output = disc(fake).reshape(-1)
    loss_gen = loss(output, torch.ones_like(output))

  optimizer_gen.zero_grad()
  g_scaler.scale(loss_gen).backward()
  g_scaler.step(optimizer_gen)
  g_scaler.update()

  return loss_disc, loss_gen

def train():
  # dataset = datasets.MNIST(root="datasets/mnist", train=True, transform=transform, download=True)
  dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS, pin_memory=True, persistent_workers=True)

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

  iteration = 0
  test_noise = torch.randn((NUMBER_OF_SAMPLE_IMAGES, NOISE_DIM, 1, 1), device=device)
  if metadata is not None:
    if "iteration" in metadata.keys():
      iteration = int(metadata["iteration"])

    if "noise" in metadata.keys():
      tmp_noise = torch.Tensor(metadata["noise"])
      if tmp_noise.shape == (NUMBER_OF_SAMPLE_IMAGES, NOISE_DIM, 1, 1):
        test_noise = tmp_noise.to(device)

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

  loss = nn.BCEWithLogitsLoss()

  summary_writer = SummaryWriter(f"logs/{MODEL_NAME}")

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  gen.train()
  disc.train()

  print("Discriminator")
  summary(disc, (IMG_CH, IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

  print("Generator")
  summary(gen, (NOISE_DIM, 1, 1), batch_size=BATCH_SIZE)

  if not os.path.exists(f"models/{MODEL_NAME}"):
    pathlib.Path(f"models/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    with tqdm(total=ITERATIONS, initial=iteration, unit="it") as bar:
      while True:
        for data, _ in loader:
          loss_disc, loss_gen = train_step(data, disc, gen, optimizer_disc, optimizer_gen, loss, d_scaler, g_scaler)
          if loss_gen is not None and loss_disc is not None:
            summary_writer.add_scalar("Gen Loss", loss_gen, global_step=iteration)
            summary_writer.add_scalar("Disc Loss", loss_disc, global_step=iteration)

          if SAVE_CHECKPOINT and iteration % CHECKPOINT_EVERY == 0:
            save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen_{iteration}.mod")
            save_model(disc, optimizer_disc, f"models/{MODEL_NAME}/disc_{iteration}.mod")

          if iteration % SAMPLE_INTERVAL == 0:
            gen.eval()
            with torch.no_grad():
              fake = gen(test_noise)
              img_grid_fake = torchvision.utils.make_grid(fake[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)
              summary_writer.add_image("Generated", img_grid_fake, global_step=iteration)
            gen.train()

          iteration += 1
          bar.update()
          if iteration >= ITERATIONS:
            break

        if iteration >= ITERATIONS:
          break

        save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
        save_model(disc, optimizer_disc, f"models/{MODEL_NAME}/disc.mod")
        save_metadata({"iteration": iteration, "noise": test_noise.tolist()}, f"models/{MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")

  save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
  save_model(disc, optimizer_disc, f"models/{MODEL_NAME}/disc.mod")
  save_metadata({"iteration": iteration, "noise": test_noise.tolist()}, f"models/{MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
    train()