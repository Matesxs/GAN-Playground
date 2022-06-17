import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from math import log2
from tqdm import tqdm
import os
import pathlib

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata
from model import Critic, Generator
import settings

from gans.utils.datasets import SingleInSingleOutDataset

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
  transform = A.Compose(
    [
      A.Resize(image_size, image_size),
      A.HorizontalFlip(p=0.5),
      A.Normalize([0.5 for _ in range(settings.IMG_CH)], [0.5 for _ in range(settings.IMG_CH)]),
      ToTensorV2()
    ]
  )

  batch_size = settings.IMG_SIZE_TO_BATCH_SIZE[image_size]
  dataset = SingleInSingleOutDataset(root_dir=settings.DATASET_PATH, transform=transform, format="RGB" if settings.IMG_CH == 3 else "GRAY")
  loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=settings.NUM_OF_WORKERS, pin_memory=True, persistent_workers=True)
  return loader, dataset

def train(data, crit, gen, step, alpha, opt_critic, opt_generator, c_scaler, g_scaler):
  real = data.to(settings.device)
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

  return loss_crit, loss_gen

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
  iteration = 0
  global_step = 0
  alpha = settings.START_ALPHA
  test_noise = torch.randn((settings.TESTING_SAMPLES, settings.Z_DIM, 1, 1), device=settings.device)
  if metadata is not None:
    if "img_size" in metadata.keys():
      start_image_size = metadata["img_size"]

    if "iteration" in metadata.keys():
      iteration = metadata["iteration"]

    if "global_step" in metadata.keys():
      global_step = metadata["global_step"]

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

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  step = (int(log2(start_image_size / 4)))
  img_size = settings.START_IMAGE_SIZE * 2 ** step
  try:
    for number_of_iterations in settings.PROGRESSIVE_ITERATIONS[step:]:
      img_size = settings.START_IMAGE_SIZE*2**step
      loader, dataset = get_loader(img_size)
      number_of_batches = len(loader)
      number_of_epochs = number_of_iterations / number_of_batches
      alpha_coef = settings.IMG_SIZE_TO_BATCH_SIZE[img_size] / ((number_of_epochs * 0.5) * len(dataset))

      print(f"Starting image size: {img_size} with batch size: {settings.IMG_SIZE_TO_BATCH_SIZE[img_size]} which coresponds to {number_of_batches} number of batches and training of {number_of_epochs} epochs")
      with tqdm(total=number_of_iterations, initial=iteration, unit="it") as bar:
        while True:
          for data in loader:
            crit_loss, gen_loss = train(data, crit, gen, step, alpha, opt_critic, opt_generator, c_scaler, g_scaler)
            alpha += alpha_coef
            alpha = min(alpha, 1)

            if gen_loss is not None:
              summary_writer.add_scalar("Gen Loss", gen_loss, global_step=global_step)
            if crit_loss is not None:
              summary_writer.add_scalar("Crit Loss", crit_loss, global_step=global_step)
            summary_writer.add_scalar("Alpha", alpha, global_step=global_step)

            if settings.SAVE_CHECKPOINTS and iteration % settings.CHECKPOINT_EVERY == 0:
              save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen_{global_step}.mod")
              save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit_{global_step}.mod")

            if iteration % settings.SAMPLE_EVERY == 0:
              print(f"Loss crit: {crit_loss:.4f}, Loss gen: {gen_loss:.4f}")

              gen.eval()

              with torch.no_grad():
                fake = gen(test_noise, alpha, step)
                img_grid_fake = torchvision.utils.make_grid(fake[:settings.TESTING_SAMPLES], normalize=True)
                summary_writer.add_image("Generated", img_grid_fake, global_step=global_step)

              gen.train()

            iteration += 1

            bar.update()
            if iteration > number_of_iterations:
              break

            global_step += 1
          if iteration > number_of_iterations:
            break

          save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
          save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit.mod")
          save_metadata({"iteration": iteration, "img_size": img_size, "global_step": global_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")
      iteration = 0
      alpha = settings.START_ALPHA
      step += 1
  except KeyboardInterrupt:
    print("Exiting")
  except Exception as e:
    print(e)
    save_metadata({"iteration": iteration, "img_size": img_size, "global_step": global_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")
    exit(-1)

  save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
  save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit.mod")
  save_metadata({"iteration": iteration, "img_size": img_size, "global_step": global_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
  main()
