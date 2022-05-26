import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from math import log2
from tqdm import tqdm
import os
import pathlib

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata
from model import Critic, Generator
import settings

def gradient_penalty(critic, real, fake, alpha, step, device):
  batch, channels, height, width = real.shape
  epsilon = torch.randn((batch, 1, 1, 1)).repeat(1, channels, height, width).to(device)
  interpolated_images = real * epsilon + fake.detach() * (1 - epsilon)
  interpolated_images.requires_grad_(True)

  interpolated_scores = critic(interpolated_images, alpha, step)
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

def get_loader(image_size):
  transform = transforms.Compose(
    [
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(0.5),
      transforms.Normalize([0.5 for _ in range(settings.IMG_CH)], [0.5 for _ in range(settings.IMG_CH)])
    ]
  )

  batch_size = settings.IMG_SIZE_TO_BATCH_SIZE[image_size]
  dataset = datasets.ImageFolder(root=settings.DATASET_PATH, transform=transform)
  loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=settings.NUM_OF_WORKERS, pin_memory=True, persistent_workers=True)
  return loader, dataset

def train(crit, gen, loader, dataset, step, alpha, opt_critic, opt_generator, c_scaler, g_scaler):
  loop = tqdm(loader, leave=True, unit="batch")

  loss_crit = loss_gen = real = None
  for batch_idx, (real, _) in enumerate(loop):
    real = real.to(settings.device)
    cur_batch_size = real.shape[0]

    # Train critic
    noise = torch.randn((cur_batch_size, settings.Z_DIM, 1, 1), device=settings.device)

    with torch.cuda.amp.autocast():
      fake = gen(noise, alpha, step)
      critic_real = crit(real, alpha, step)
      critic_fake = crit(fake.detach(), alpha, step)

      gp = gradient_penalty(crit, real, fake, alpha, step, device=settings.device)
      loss_crit = -(torch.mean(critic_real) - torch.mean(critic_fake)) + settings.LAMBDA_GP * gp + (0.001 * torch.mean(critic_real ** 2))

    opt_critic.zero_grad()
    c_scaler.scale(loss_crit).backward()
    c_scaler.step(opt_critic)
    c_scaler.update()

    # Train generator
    with torch.cuda.amp.autocast():
      gen_fake = crit(fake, alpha, step)
      loss_gen = -torch.mean(gen_fake)

    opt_generator.zero_grad()
    g_scaler.scale(loss_gen).backward()
    g_scaler.step(opt_generator)
    g_scaler.update()

    alpha += cur_batch_size / (len(dataset) * settings.PROGRESSIVE_EPOCHS[step] * 0.5)
    alpha = min(alpha, 1)

  return loss_crit, loss_gen, alpha, real

def main():
  crit = Critic(settings.FEATURES, settings.IMG_CH).to(settings.device)
  gen = Generator(settings.Z_DIM, settings.FEATURES, settings.IMG_CH).to(settings.device)

  opt_generator = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0.0, 0.99))
  opt_critic = optim.Adam(crit.parameters(), lr=settings.LR, betas=(0.0, 0.99))

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/crit.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/crit.mod"):
      load_model(f"models/{settings.MODEL_NAME}/crit.mod", crit, opt_critic, settings.LR, settings.device)
  except:
    print("Critic model is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/gen.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/gen.mod"):
      load_model(f"models/{settings.MODEL_NAME}/gen.mod", gen, opt_generator, settings.LR, settings.device)
  except:
    print("Generator model is incompatible with found model parameters")

  metadata = None
  if os.path.exists(f"models/{settings.MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{settings.MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{settings.MODEL_NAME}/metadata.pkl")

  start_image_size = settings.START_IMAGE_SIZE
  start_epoch = 0
  tensorboard_step = 0
  alpha = settings.START_ALPHA
  test_noise = torch.randn((settings.TESTING_SAMPLES, settings.Z_DIM, 1, 1), device=settings.device)
  if metadata is not None:
    if "img_size" in metadata.keys():
      start_image_size = metadata["img_size"]

    if "epoch" in metadata.keys():
      start_epoch = metadata["epoch"]

    if "tbstep" in metadata.keys():
      tensorboard_step = metadata["tbstep"]

    if "alpha" in metadata.keys():
      alpha = metadata["alpha"]

    if "noise" in metadata.keys():
      tmp_noise = torch.Tensor(metadata["noise"])
      if tmp_noise.shape == (settings.TESTING_SAMPLES, settings.Z_DIM, 1, 1):
        test_noise = tmp_noise.to(settings.device)

  if settings.GEN_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.GEN_MODEL_WEIGHTS_TO_LOAD, gen, opt_generator, settings.LR, settings.device)
    except Exception as e:
      print(f"Generator model weights are incompatible with found model parameters\n{e}")
      exit(2)

  if settings.CRIT_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.CRIT_MODEL_WEIGHTS_TO_LOAD, crit, opt_critic, settings.LR, settings.device)
    except Exception as e:
      print(f"Critic model weights are incompatible with found model parameters\n{e}")
      exit(2)

  g_scaler = torch.cuda.amp.GradScaler()
  c_scaler = torch.cuda.amp.GradScaler()

  img_size = settings.START_IMAGE_SIZE
  epoch = start_epoch

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  step = (int(log2(start_image_size / 4))) if settings.STEP_OVERRIDE is None else settings.STEP_OVERRIDE
  try:
    for epochs_idx, num_epochs in enumerate(settings.PROGRESSIVE_EPOCHS[step:]):
      img_size = 4*2**step
      loader, dataset = get_loader(img_size)
      print(f"Starting image size: {img_size} with batch size: {settings.IMG_SIZE_TO_BATCH_SIZE[img_size]}")

      for epoch in range(start_epoch, num_epochs):
        crit_loss, gen_loss, alpha, last_real = train(crit, gen, loader, dataset, step, alpha, opt_critic, opt_generator, c_scaler, g_scaler)

        if crit_loss is not None and gen_loss is not None:
          print(f"PH: {step}/{settings.NUM_OF_STEPES}  Epoch: {epoch}/{num_epochs} Loss crit: {crit_loss:.4f}, Loss gen: {gen_loss:.4f}")
          summary_writer.add_scalar("Gen Loss", gen_loss, global_step=tensorboard_step)
          summary_writer.add_scalar("Critic Loss", crit_loss, global_step=tensorboard_step)
          summary_writer.add_scalar("Alpha", alpha, global_step=tensorboard_step)

        if settings.SAVE_CHECKPOINTS and epoch % settings.CHECKPOINT_EVERY == 0:
          save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen_{step}_{epoch}.mod")
          save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit_{step}_{epoch}.mod")

        save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
        save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit.mod")

        if epoch % settings.SAMPLE_EVERY == 0:
          gen.eval()

          with torch.no_grad():
            fake = gen(test_noise, alpha, step)

            img_grid_fake = torchvision.utils.make_grid(fake[:settings.TESTING_SAMPLES], normalize=True)
            img_grid_real = torchvision.utils.make_grid(last_real[:settings.TESTING_SAMPLES], normalize=True)

            summary_writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)
            summary_writer.add_image("Real", img_grid_real, global_step=tensorboard_step)

          gen.train()

        tensorboard_step += 1
        save_metadata({"epoch": epoch, "img_size": img_size, "tbstep": tensorboard_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")

      start_epoch = 0
      step += 1
      alpha = settings.START_ALPHA
  except KeyboardInterrupt:
    print("Exiting")
    pass

  save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
  save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit.mod")
  save_metadata({"epoch": epoch, "img_size": img_size, "tbstep": tensorboard_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
  main()
