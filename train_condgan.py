import os
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pathlib
from tqdm import tqdm

from gans.Conditional_GAN.critic_model import Critic
from gans.Conditional_GAN.generator_model import Generator
from gans.Conditional_GAN.settings import *

from gans.utils.training_saver import load_model, save_model, load_metadata, save_metadata

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def gradient_penalty(critic, real, fake, labels, device):
  batch, channels, height, width = real.shape
  epsilon = torch.rand((batch, 1, 1, 1)).repeat(1, channels, height, width).to(device)
  interpolated_imgs = real * epsilon + fake.detach() * (1 - epsilon)
  interpolated_imgs.requires_grad_(True)

  interpolated_scores = critic(interpolated_imgs, labels)
  gradient = torch.autograd.grad(inputs=interpolated_imgs, outputs=interpolated_scores,
                                 grad_outputs=torch.ones_like(interpolated_scores),
                                 create_graph=True, retain_graph=True)[0]

  # Flattening of gradient
  gradient = gradient.view(gradient.shape[0], -1)
  gradient_norm = gradient.norm(2, dim=1) # L2 norm

  return torch.mean((gradient_norm - 1) ** 2)

def train_step(real, labels, crit, gen, optimizer_crit, optimizer_gen, step):
  real = real.to(device)
  labels = labels.to(device)

  # Train critic
  noise = torch.randn((labels.shape[0], NOISE_DIM, 1, 1), device=device)
  fake = gen(noise, labels)
  crit_real = crit(real, labels).reshape(-1)  # Flatten
  crit_fake = crit(fake.detach(), labels).reshape(-1)

  gp = gradient_penalty(crit, real, fake, labels, device)
  loss_crit = -(torch.mean(crit_real) - torch.mean(crit_fake)) + LAMBDA_GRAD_PENALTY * gp

  crit.zero_grad()
  loss_crit.backward()
  optimizer_crit.step()

  # Train generator
  loss_gen = None
  if step % CRITIC_ITERATIONS == 0:
    output = crit(fake, labels).reshape(-1)
    loss_gen = -torch.mean(output)

    gen.zero_grad()
    loss_gen.backward()
    optimizer_gen.step()

  return loss_crit, loss_gen

def train():
  dataset = datasets.MNIST(root="datasets/mnist", train=True, transform=transform, download=True)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS, persistent_workers=True, pin_memory=True)

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

  iteration = 0
  test_noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1), device=device)
  if metadata is not None:
    if "iteration" in metadata.keys():
      iteration = int(metadata["iteration"])

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

  summary_writer = SummaryWriter(f"logs/{MODEL_NAME}")

  gen.train()
  crit.train()

  if not os.path.exists(f"models/{MODEL_NAME}"):
    pathlib.Path(f"models/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    with tqdm(total=ITERATIONS, initial=iteration, unit="it") as bar:
      while True:
        for real, labels in loader:
          loss_crit, loss_gen = train_step(real, labels, crit, gen, optimizer_crit, optimizer_gen, iteration)
          if loss_gen is not None:
            summary_writer.add_scalar("Gen Loss", loss_gen, global_step=iteration)
          if loss_crit is not None:
            summary_writer.add_scalar("Crit Loss", loss_crit, global_step=iteration)

          if SAVE_CHECKPOINT and iteration % CHECKPOINT_EVERY == 0:
            save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen_{iteration}.mod")
            save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit_{iteration}.mod")

          if iteration % SAMPLE_EVERY == 0:
            gen.eval()
            with torch.no_grad():
              fake = gen(test_noise, labels.to(device))

              img_grid_real = torchvision.utils.make_grid(real[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)
              img_grid_fake = torchvision.utils.make_grid(fake[:NUMBER_OF_SAMPLE_IMAGES], normalize=True)

              summary_writer.add_image("Real", img_grid_real, global_step=iteration)
              summary_writer.add_image("Fake", img_grid_fake, global_step=iteration)
            gen.train()

          bar.update()
          iteration += 1
          if iteration > ITERATIONS:
            break

        if iteration > ITERATIONS:
          break

        save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
        save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit.mod")
        save_metadata({"iteration": iteration, "noise": test_noise.tolist()}, f"models/{MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")

  save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen.mod")
  save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit.mod")
  save_metadata({"iteration": iteration, "noise": test_noise.tolist()}, f"models/{MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
    train()