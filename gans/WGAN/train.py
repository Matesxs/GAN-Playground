import os
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import pathlib
from tqdm import tqdm

from critic_model import Critic
from generator_model import Generator
from settings import *

from gans.utils.training_saver import load_model, save_model, load_metadata, save_metadata

transform = transforms.Compose(
  [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CH)], [0.5 for _ in range(IMG_CH)])
  ]
)

def train_step(data, crit, gen, optimizer_crit, optimizer_gen, c_scaler, g_scaler):
  real = data.to(device)
  noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1), device=device)

  # Train critic
  for _ in range(CRITIC_ITERATIONS):
    with torch.cuda.amp.autocast():
      fake = gen(noise)

      crit_real = crit(real).reshape(-1)  # Flatten
      crit_fake = crit(fake).reshape(-1)

      loss_crit = -(torch.mean(crit_real) - torch.mean(crit_fake))

    optimizer_crit.zero_grad()
    c_scaler.scale(loss_crit).backward()
    c_scaler.step(optimizer_crit)
    c_scaler.update()

    for p in crit.parameters():
      p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

    noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1), device=device)

  # Train generator
  with torch.cuda.amp.autocast():
    fake = gen(noise)
    output = crit(fake).reshape(-1)
    loss_gen = -torch.mean(output)

  optimizer_gen.zero_grad()
  g_scaler.scale(loss_gen).backward()
  g_scaler.step(optimizer_gen)
  g_scaler.update()

  return loss_crit, loss_gen

def train():
  # dataset = datasets.MNIST(root="datasets/mnist", train=True, transform=transform, download=True)
  dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS, persistent_workers=True, pin_memory=True)

  gen = Generator(NOISE_DIM, IMG_CH, FEATURES_GEN).to(device)
  crit = Critic(IMG_CH, FEATURES_CRIT).to(device)

  optimizer_gen = optim.RMSprop(gen.parameters(), lr=LR)
  optimizer_crit = optim.RMSprop(crit.parameters(), lr=LR)

  try:
    if os.path.exists(f"models/{MODEL_NAME}/crit.mod") and os.path.isfile(f"models/{MODEL_NAME}/crit.mod"):
      load_model(f"models/{MODEL_NAME}/crit.mod", crit, optimizer_crit, LR, device)
  except:
    print("Critic model is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{MODEL_NAME}/gen.mod") and os.path.isfile(f"models/{MODEL_NAME}/gen.mod"):
      load_model(f"models/{MODEL_NAME}/gen.mod", gen, optimizer_gen, LR, device)
  except:
    print("Generator model is incompatible with found model parameters")

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

  if CRITIC_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(CRITIC_MODEL_WEIGHTS_TO_LOAD, crit, optimizer_crit, LR, device)
    except:
      print("Critic model weights are incompatible with found model parameters")
      exit(2)

  summary_writer = SummaryWriter(f"logs/{MODEL_NAME}")

  g_scaler = torch.cuda.amp.GradScaler()
  c_scaler = torch.cuda.amp.GradScaler()

  gen.train()
  crit.train()

  print("Critic")
  summary(crit, (IMG_CH, IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

  print("Generator")
  summary(gen, (NOISE_DIM, 1, 1), batch_size=BATCH_SIZE)

  if not os.path.exists(f"models/{MODEL_NAME}"):
    pathlib.Path(f"models/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    with tqdm(total=ITERATIONS, initial=iteration, unit="it") as bar:
      while True:
        for data, _ in loader:
          loss_crit, loss_gen = train_step(data, crit, gen, optimizer_crit, optimizer_gen, c_scaler, g_scaler)
          if loss_crit is not None and loss_gen is not None:
            summary_writer.add_scalar("Gen Loss", loss_gen, global_step=iteration)
            summary_writer.add_scalar("Crit Loss", loss_crit, global_step=iteration)

          if SAVE_CHECKPOINT and iteration % CHECKPOINT_EVERY == 0:
            save_model(gen, optimizer_gen, f"models/{MODEL_NAME}/gen_{iteration}.mod")
            save_model(crit, optimizer_crit, f"models/{MODEL_NAME}/crit_{iteration}.mod")

          if iteration % SAMPLE_EVERY == 0:
            gen.eval()
            with torch.no_grad():
              fake = gen(test_noise)

              img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

              summary_writer.add_image("Generated", img_grid_fake, global_step=iteration)
            gen.train()

          bar.update()
          iteration += 1
          if iteration >= ITERATIONS:
            break

        if iteration >= ITERATIONS:
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