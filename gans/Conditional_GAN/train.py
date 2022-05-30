import os
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pathlib
from tqdm import tqdm

from critic_model import Critic
from generator_model import Generator
from helpers import gradient_penalty
from settings import *

from gans.utils.training_saver import load_model, save_model, load_metadata, save_metadata

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def train_step(loader, crit, gen, optimizer_crit, optimizer_gen):
  loop = tqdm(loader, leave=True, unit="batch")

  loss_crit = loss_gen = labels = real = None
  for batch_idx, (real, labels) in enumerate(loop):
    real = real.to(device)
    labels = labels.to(device)

    noise = torch.randn((labels.shape[0], NOISE_DIM, 1, 1), device=device)
    fake = gen(noise, labels)

    # Train critic
    for _ in range(CRITIC_ITERATIONS):
      crit_real = crit(real, labels).reshape(-1)  # Flatten
      crit_fake = crit(fake, labels).reshape(-1)

      gp = gradient_penalty(crit, real, fake, labels, device)
      loss_crit = -(torch.mean(crit_real) - torch.mean(crit_fake)) + LAMBDA_GRAD_PENALTY * gp

      crit.zero_grad()
      loss_crit.backward(retain_graph=True)
      optimizer_crit.step()

      noise = torch.randn((labels.shape[0], NOISE_DIM, 1, 1), device=device)
      fake = gen(noise, labels)

    # Train generator
    output = crit(fake, labels).reshape(-1)
    loss_gen = -torch.mean(output)
    gen.zero_grad()
    loss_gen.backward()
    optimizer_gen.step()

  return loss_crit, loss_gen, labels, real

def train():
  dataset = datasets.MNIST(root="datasets/mnist", train=True, transform=transform, download=True)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

  gen = Generator(NOISE_DIM, IMG_CH, FEATURES_GEN, NUM_OF_CLASSES, IMG_SIZE, EMBED_SIZE).to(device)
  crit = Critic(IMG_CH, FEATURES_CRIT, NUM_OF_CLASSES, IMG_SIZE).to(device)

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
  test_noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1), device=device)
  if metadata is not None:
    if "epoch" in metadata.keys():
      start_epoch = int(metadata["epoch"])

    if "noise" in metadata.keys():
      tmp_noise = torch.Tensor(metadata["noise"])
      if tmp_noise.shape == (BATCH_SIZE, NOISE_DIM, 1, 1):
        test_noise = tmp_noise.to(device)

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

  summary_writer_real = SummaryWriter(f"logs/{MODEL_NAME}/real")
  summary_writer_fake = SummaryWriter(f"logs/{MODEL_NAME}/fake")
  summary_writer_values = SummaryWriter(f"logs/{MODEL_NAME}/scalars")

  gen.train()
  crit.train()

  if not os.path.exists(f"models/{MODEL_NAME}"):
    pathlib.Path(f"models/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  last_epoch = 0
  try:
    for epoch in range(start_epoch, EPOCHS):
      last_epoch = epoch

      loss_crit, loss_gen, labels, real = train_step(loader, crit, gen, optimizer_crit, optimizer_gen)
      if loss_crit is not None and loss_gen is not None:
        print(f"Epoch: {epoch}/{EPOCHS} Loss crit: {loss_crit:.4f}, Loss gen: {loss_gen:.4f}")
        summary_writer_values.add_scalar("Gen Loss", loss_gen, global_step=epoch)
        summary_writer_values.add_scalar("Crit Loss", loss_crit, global_step=epoch)

      if epoch % SAMPLE_EVERY == 0:
        with torch.no_grad():
          fake = gen(test_noise, labels)

          img_grid_real = torchvision.utils.make_grid(real[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)
          img_grid_fake = torchvision.utils.make_grid(fake[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)

          summary_writer_real.add_image("Real", img_grid_real, global_step=epoch)
          summary_writer_fake.add_image("Fake", img_grid_fake, global_step=epoch)

        save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen_{epoch}.mod")
        save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit_{epoch}.mod")

      save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
      save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit.mod")
      save_metadata({"epoch": last_epoch, "noise": test_noise.tolist()}, f"models/{MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")

  save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
  save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit.mod")
  save_metadata({"epoch": last_epoch, "noise": test_noise.tolist()}, f"models/{MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
    train()