import os
import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import Critic, Generator
from helpers import gradient_penalty

LR = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
IMG_CH = 1
NUM_OF_CLASSES = 10
NOISE_DIM = 128
EMBED_SIZE = 128

EPOCHS = 20
FEATURES_CRIT = FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GRAD_PENALTY = 10

START_STEP_VAL = 0
SAMPLE_PER_STEPS = 100

MODEL_NAME = "mnist_cond-gan_model"
NUMBER_OF_SAMPLE_IMAGES = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def train():
  dataset = datasets.MNIST(root="datasets/mnist", train=True, transform=transform, download=True)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

  gen = Generator(NOISE_DIM, IMG_CH, FEATURES_GEN, NUM_OF_CLASSES, IMG_SIZE, EMBED_SIZE).to(device)
  crit = Critic(IMG_CH, FEATURES_CRIT, NUM_OF_CLASSES, IMG_SIZE).to(device)

  try:
    if os.path.exists(f"models/{MODEL_NAME}_gen.mod") and os.path.isfile(f"models/{MODEL_NAME}_gen.mod"):
      gen.load_state_dict(torch.load(f"models/{MODEL_NAME}_gen.mod"))

    if os.path.exists(f"models/{MODEL_NAME}_crit.mod") and os.path.isfile(f"models/{MODEL_NAME}_crit.mod"):
      crit.load_state_dict(torch.load(f"models/{MODEL_NAME}_crit.mod"))
  except:
    print("Models are incompatible with found model parameters")
    exit(1)

  optimizer_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.9))
  optimizer_crit = optim.Adam(crit.parameters(), lr=LR, betas=(0.5, 0.9))

  test_noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1), device=device)

  summary_writer_real = SummaryWriter(f"logs/{MODEL_NAME}/real")
  summary_writer_fake = SummaryWriter(f"logs/{MODEL_NAME}/fake")
  summary_writer_values = SummaryWriter(f"logs/{MODEL_NAME}/scalars")
  step = START_STEP_VAL

  gen.train()
  crit.train()

  if not os.path.exists("models"):
    os.mkdir("models")

  try:
    for epoch in range(EPOCHS):
      for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        labels = labels.to(device)

        noise = torch.randn((labels.shape[0], NOISE_DIM, 1, 1), device=device)
        fake = gen(noise, labels)

        # Train critic
        loss_crit = None
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

        if batch_idx % SAMPLE_PER_STEPS == 0:
          print(f"Epoch: {epoch}/{EPOCHS} Batch: {batch_idx}/{len(loader)} Loss crit: {loss_crit:.4f}, Loss gen: {loss_gen:.4f}")

          with torch.no_grad():
            fake = gen(test_noise, labels)

            img_grid_real = torchvision.utils.make_grid(real[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)

            summary_writer_real.add_image("Real", img_grid_real, global_step=step)
            summary_writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            summary_writer_values.add_scalar("Gen Loss", loss_gen, global_step=step)
            summary_writer_values.add_scalar("Critic Loss", loss_crit, global_step=step)

          step += 1

        if batch_idx % 500 == 0:
          torch.save(gen.state_dict(), f"models/{MODEL_NAME}_gen.mod")
          torch.save(crit.state_dict(), f"models/{MODEL_NAME}_crit.mod")
  except KeyboardInterrupt:
    print("Exiting")

  torch.save(gen.state_dict(), f"models/{MODEL_NAME}_gen.mod")
  torch.save(crit.state_dict(), f"models/{MODEL_NAME}_crit.mod")

if __name__ == '__main__':
    train()