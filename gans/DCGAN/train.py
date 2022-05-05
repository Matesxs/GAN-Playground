import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models import Discriminator, Generator

LR = 2e-4
BATCH_SIZE = 128
IMG_SIZE = 128
IMG_CH = 3
NOISE_DIM = 128

EPOCHS = 20
FEATURES_DISC = FEATURES_GEN = 64

START_STEP_VAL = 0

MODEL_NAME = "celeb_BIG_dcgan_model"
NUMBER_OF_SAMPLE_IMAGES = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def train():
  # dataset = datasets.MNIST(root="datasets/mnist", train=True, transform=transform, download=True)
  # dataset = datasets.ImageFolder(root="datasets/celeb_normalized__64x64", transform=transform)
  dataset = datasets.ImageFolder(root="datasets/celeb_normalized__128x128", transform=transform)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

  gen = Generator(NOISE_DIM, IMG_CH, FEATURES_GEN).to(device)
  disc = Discriminator(IMG_CH, FEATURES_DISC).to(device)

  try:
    if os.path.exists(f"models/{MODEL_NAME}_gen.mod") and os.path.isfile(f"models/{MODEL_NAME}_gen.mod"):
      gen.load_state_dict(torch.load(f"models/{MODEL_NAME}_gen.mod"))

    if os.path.exists(f"models/{MODEL_NAME}_disc.mod") and os.path.isfile(f"models/{MODEL_NAME}_disc.mod"):
      disc.load_state_dict(torch.load(f"models/{MODEL_NAME}_disc.mod"))
  except:
    print("Models are incompatible with found model parameters")
    exit(1)

  optimizer_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
  optimizer_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

  loss = nn.BCELoss()

  test_noise = torch.randn((32, NOISE_DIM, 1, 1), device=device)

  summary_writer_real = SummaryWriter(f"logs/{MODEL_NAME}/real")
  summary_writer_fake = SummaryWriter(f"logs/{MODEL_NAME}/fake")
  step = START_STEP_VAL

  gen.train()
  disc.train()

  print("Discriminator")
  summary(disc, (IMG_CH, IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

  print("Generator")
  summary(gen, (NOISE_DIM, 1, 1), batch_size=BATCH_SIZE)

  if not os.path.exists("models"):
    os.mkdir("models")

  try:
    for epoch in range(EPOCHS):
      for batch_idx, (real, _) in enumerate(loader):
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

        if batch_idx % 100 == 0:
          print(f"Epoch: {epoch}/{EPOCHS} Batch: {batch_idx}/{len(loader)} Loss disc: {loss_disc:.4f}, Loss gen: {loss_gen:.4f}")

          with torch.no_grad():
            fake = gen(test_noise)

            img_grid_real = torchvision.utils.make_grid(real[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)

            summary_writer_real.add_image("Real", img_grid_real, global_step=step)
            summary_writer_fake.add_image("Fake", img_grid_fake, global_step=step)

          step += 1
  except KeyboardInterrupt:
    print("Exiting")

  torch.save(gen.state_dict(), f"models/{MODEL_NAME}_gen.mod")
  torch.save(disc.state_dict(), f"models/{MODEL_NAME}_disc.mod")

if __name__ == '__main__':
    train()