import os
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import pathlib

from critic_model import Critic
from generator_model import Generator
from helpers import gradient_penalty
from settings import *

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def train():
  # dataset = datasets.MNIST(root="datasets/mnist", train=True, transform=transform, download=True)
  dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

  gen = Generator(NOISE_DIM, IMG_CH, FEATURES_GEN).to(device)
  crit = Critic(IMG_CH, FEATURES_CRIT).to(device)

  optimizer_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.9))
  optimizer_crit = optim.Adam(crit.parameters(), lr=LR, betas=(0.5, 0.9))

  try:
    if os.path.exists(f"models/{MODEL_NAME}/gen.mod") and os.path.isfile(f"models/{MODEL_NAME}/gen.mod"):
      load_model(f"models/{MODEL_NAME}/gen.mod", gen, optimizer_gen, LR, device)

    if os.path.exists(f"models/{MODEL_NAME}/crit.mod") and os.path.isfile(f"models/{MODEL_NAME}/crit.mod"):
      load_model(f"models/{MODEL_NAME}/crit.mod", crit, optimizer_crit, LR, device)
  except:
    print("Models are incompatible with found model parameters")

  metadata = None
  if os.path.exists(f"models/{MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{MODEL_NAME}/metadata.pkl")

  start_epoch = 0
  start_stepval = 0
  if metadata is not None:
    if "epoch" in metadata.keys():
      start_epoch = int(metadata["epoch"])

    if "stepval" in metadata.keys():
      start_stepval = int(metadata["stepval"])

  if GEN_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(GEN_MODEL_WEIGHTS_TO_LOAD, gen, optimizer_gen, LR, device)
    except:
      print("Generator model weights are incompatible with found model parameters")
      exit(2)

  if CRITIC_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(CRITIC_MODEL_WEIGHTS_TO_LOAD, crit, optimizer_crit, LR, device)
    except:
      print("Critic model weights are incompatible with found model parameters")
      exit(2)

  test_noise = torch.randn((32, NOISE_DIM, 1, 1), device=device)

  summary_writer_real = SummaryWriter(f"logs/{MODEL_NAME}/real")
  summary_writer_fake = SummaryWriter(f"logs/{MODEL_NAME}/fake")
  summary_writer_values = SummaryWriter(f"logs/{MODEL_NAME}/scalars")
  step = start_stepval

  gen.train()
  crit.train()

  print("Critic")
  summary(crit, (IMG_CH, IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

  print("Generator")
  summary(gen, (NOISE_DIM, 1, 1), batch_size=BATCH_SIZE)

  if not os.path.exists(f"models/{MODEL_NAME}"):
    pathlib.Path(f"models/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  last_epoch = 0
  try:
    for epoch in range(start_epoch, EPOCHS):
      last_epoch = epoch

      for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((real.shape[0], NOISE_DIM, 1, 1), device=device)
        fake = gen(noise)

        # Train critic
        loss_crit = None
        for _ in range(CRITIC_ITERATIONS):
          crit_real = crit(real).reshape(-1)  # Flatten
          crit_fake = crit(fake).reshape(-1)

          gp = gradient_penalty(crit, real, fake, device)
          loss_crit = -(torch.mean(crit_real) - torch.mean(crit_fake)) + LAMBDA_GRAD_PENALTY * gp

          crit.zero_grad()
          loss_crit.backward(retain_graph=True)
          optimizer_crit.step()

          noise = torch.randn((real.shape[0], NOISE_DIM, 1, 1), device=device)
          fake = gen(noise)

        # Train generator
        output = crit(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch_idx % SAMPLE_PER_STEPS == 0:
          print(f"Epoch: {epoch}/{EPOCHS} Batch: {batch_idx}/{len(loader)} Loss crit: {loss_crit:.4f}, Loss gen: {loss_gen:.4f}")

          with torch.no_grad():
            fake = gen(test_noise)

            img_grid_real = torchvision.utils.make_grid(real[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)

            summary_writer_real.add_image("Real", img_grid_real, global_step=step)
            summary_writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            summary_writer_values.add_scalar("Gen Loss", loss_gen, global_step=step)
            summary_writer_values.add_scalar("Critic Loss", loss_crit, global_step=step)

          save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen_{step}.mod")
          save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit_{step}.mod")

          save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
          save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit.mod")

          step += 1
          save_metadata({"epoch": last_epoch, "stepval": step}, f"models/{MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")
    save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
    save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit.mod")
    save_metadata({"epoch": last_epoch, "stepval": step}, f"models/{MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
    train()