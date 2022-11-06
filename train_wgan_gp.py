import os
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import pathlib
from tqdm import tqdm
import traceback
import numpy as np

from gans.WGAN_GP.critic_model import Critic
from gans.WGAN_GP.generator_model import Generator
from gans.WGAN_GP import settings

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata
from gans.utils.datasets import SingleInSingleOutDataset
from gans.utils.learning import get_linear_lr_decay_scheduler

torch.backends.cudnn.benchmark = True

def gradient_penalty(critic, real, fake, device):
  batch, channels, height, width = real.shape
  epsilon = torch.rand((batch, 1, 1, 1)).repeat(1, channels, height, width).to(device)
  interpolated_imgs = real * epsilon + fake.detach() * (1 - epsilon)
  interpolated_imgs.requires_grad_(True)

  interpolated_scores = critic(interpolated_imgs)
  gradient = torch.autograd.grad(inputs=interpolated_imgs,
                                 outputs=interpolated_scores,
                                 grad_outputs=torch.ones_like(interpolated_scores),
                                 create_graph=True,
                                 retain_graph=True,
                                 only_inputs=True)[0]

  # Flattening of gradient
  gradient = gradient.view(gradient.shape[0], -1)
  gradient_norm = gradient.norm(2, dim=1) # L2 norm

  return torch.mean((gradient_norm - 1) ** 2)

def train_step(data, crit, gen, optimizer_crit, optimizer_gen):
  real = data.to(settings.device)
  cur_batch_size = real.shape[0]

  # Train critic
  crit_losses = []
  for _ in range(settings.CRITIC_ITERATIONS):
    noise = torch.randn(cur_batch_size, settings.NOISE_DIM,).to(settings.device)
    fake = gen(noise)
    critic_real = crit(real).reshape(-1)
    critic_fake = crit(fake.detach()).reshape(-1)

    gp = gradient_penalty(crit, real, fake, device=settings.device)
    loss_crit = -(torch.mean(critic_real) - torch.mean(critic_fake)) + settings.LAMBDA_GRAD_PENALTY * gp

    crit.zero_grad()
    loss_crit.backward()
    optimizer_crit.step()
    crit_losses.append(loss_crit.item())

  # Train generator
  gen_losses = []
  for _ in range(settings.GENERATOR_ITERATIONS):
    noise = torch.randn(cur_batch_size, settings.NOISE_DIM, ).to(settings.device)
    fake = gen(noise)
    gen_fake = crit(fake).reshape(-1)
    loss_gen = -torch.mean(gen_fake)

    gen.zero_grad()
    loss_gen.backward()
    optimizer_gen.step()
    gen_losses.append(loss_gen.item())

  return np.mean(crit_losses) if len(crit_losses) > 1 else crit_losses[0], np.mean(gen_losses) if len(gen_losses) > 1 else gen_losses[0]

def train():
  dataset = SingleInSingleOutDataset(root_dir=settings.DATASET_PATH, transform=settings.transform, format="RGB" if settings.IMG_CH == 3 else "GRAY")
  loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_OF_WORKERS, persistent_workers=True, pin_memory=True)

  dataset_length = len(dataset)
  number_of_batches = dataset_length // settings.BATCH_SIZE
  number_of_epochs = settings.ITERATIONS // number_of_batches

  print(f"Found {dataset_length} training images in {number_of_batches} batches")
  print(f"Which corespondes to {number_of_epochs} epochs")

  gen = Generator(settings.NOISE_DIM, settings.IMG_CH, settings.FEATURES_GEN, settings.SCALE_OR_STRIDE_FACTOR_GEN, settings.GENERATOR_USE_PIXELSHUFFLE).to(settings.device)
  crit = Critic(settings.IMG_CH, settings.FEATURES_CRIT, settings.STRIDES_CRIT).to(settings.device)

  optimizer_gen = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0, 0.9))
  optimizer_crit = optim.Adam(crit.parameters(), lr=settings.LR, betas=(0, 0.9))

  try:
    if os.path.exists(f"models/wgan_gp/{settings.MODEL_NAME}/crit.mod") and os.path.isfile(f"models/wgan_gp/{settings.MODEL_NAME}/crit.mod"):
      load_model(f"models/wgan_gp/{settings.MODEL_NAME}/crit.mod", crit, optimizer_crit, settings.LR, settings.device)
  except:
    print("Critic model is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/wgan_gp/{settings.MODEL_NAME}/gen.mod") and os.path.isfile(f"models/wgan_gp/{settings.MODEL_NAME}/gen.mod"):
      load_model(f"models/wgan_gp/{settings.MODEL_NAME}/gen.mod", gen, optimizer_gen, settings.LR, settings.device)
  except:
    print("Generator model is incompatible with found model parameters")

  metadata = None
  if os.path.exists(f"models/wgan_gp/{settings.MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/wgan_gp/{settings.MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/wgan_gp/{settings.MODEL_NAME}/metadata.pkl")

  iteration = 0
  if metadata is not None:
    if "iteration" in metadata.keys():
      iteration = int(metadata["iteration"])

    if "noise" in metadata.keys():
      tmp_noise = torch.Tensor(metadata["noise"])
      if tmp_noise.shape == (settings.NUMBER_OF_SAMPLE_IMAGES, settings.NOISE_DIM, 1, 1):
        test_noise = tmp_noise.to(settings.device)
      else:
        test_noise = torch.randn((settings.NUMBER_OF_SAMPLE_IMAGES, settings.NOISE_DIM,), device=settings.device)
    else:
      test_noise = torch.randn((settings.NUMBER_OF_SAMPLE_IMAGES, settings.NOISE_DIM,), device=settings.device)
  else:
    test_noise = torch.randn((settings.NUMBER_OF_SAMPLE_IMAGES, settings.NOISE_DIM,), device=settings.device)

  schedulers = []
  if settings.DECAY_LR:
    schedulers.append(get_linear_lr_decay_scheduler(optimizer_gen, settings.DECAY_AFTER_ITERATIONS, settings.DECAY_ITERATION, iteration + 1))
    schedulers.append(get_linear_lr_decay_scheduler(optimizer_crit, settings.DECAY_AFTER_ITERATIONS, settings.DECAY_ITERATION, iteration + 1))

  if settings.GEN_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.GEN_MODEL_WEIGHTS_TO_LOAD, gen, optimizer_gen, settings.LR, settings.device)
    except:
      print("Generator model weights are incompatible with found model parameters")
      exit(2)

  if settings.CRITIC_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.CRITIC_MODEL_WEIGHTS_TO_LOAD, crit, optimizer_crit, settings.LR, settings.device)
    except:
      print("Critic model weights are incompatible with found model parameters")
      exit(2)

  summary_writer = SummaryWriter(f"logs/wgan_gp/{settings.MODEL_NAME}")

  gen.train()
  crit.train()

  print("Critic")
  summary(crit, (settings.IMG_CH, settings.IMG_SIZE, settings.IMG_SIZE), batch_size=settings.BATCH_SIZE)

  print("Generator")
  summary(gen, (settings.NOISE_DIM,), batch_size=settings.BATCH_SIZE)

  if not os.path.exists(f"models/wgan_gp/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/wgan_gp/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    with tqdm(total=settings.ITERATIONS, initial=iteration, unit="it") as bar:
      while True:
        for data in loader:
          loss_crit, loss_gen = train_step(data, crit, gen, optimizer_crit, optimizer_gen)

          summary_writer.add_scalar("Crit Loss", loss_crit, global_step=iteration)
          summary_writer.add_scalar("Gen Loss", loss_gen, global_step=iteration)
          summary_writer.add_scalar("LR", optimizer_gen.param_groups[0]['lr'], global_step=iteration)

          bar.set_description(f"Loss crit: {loss_crit:.4f}, Loss gen: {loss_gen:.4f}, Lr: {optimizer_gen.param_groups[0]['lr']:.7f}", refresh=False)

          if settings.SAVE_CHECKPOINT and iteration % settings.CHECKPOINT_EVERY == 0:
            save_model(gen, optimizer_gen, f"models/wgan_gp/{settings.MODEL_NAME}/gen_{iteration}.mod")
            save_model(crit, optimizer_crit, f"models/wgan_gp/{settings.MODEL_NAME}/crit_{iteration}.mod")

          if iteration % settings.SAMPLE_EVERY == 0:
            gen.eval()
            with torch.no_grad():
              eval_fake = gen(test_noise)
              eval_crit_results = crit(eval_fake).reshape(-1)

              img_grid_fake = torchvision.utils.make_grid(eval_fake[:settings.NUMBER_OF_SAMPLE_IMAGES], normalize=True)
              summary_writer.add_image("Generated", img_grid_fake, global_step=iteration)
              summary_writer.add_scalar("Mean Crit Val", torch.mean(eval_crit_results), global_step=iteration)
            gen.train()

          bar.update()
          iteration += 1

          for scheduler in schedulers:
            scheduler.step()

          if iteration > settings.ITERATIONS:
            break

        if iteration > settings.ITERATIONS:
          break

        save_model(gen, optimizer_gen, f"models/wgan_gp/{settings.MODEL_NAME}/gen.mod")
        save_model(crit, optimizer_crit, f"models/wgan_gp/{settings.MODEL_NAME}/crit.mod")
        save_metadata({"iteration": iteration, "noise": test_noise.tolist()}, f"models/wgan_gp/{settings.MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")
  except Exception:
    print(traceback.format_exc())
    save_metadata({"iteration": iteration, "noise": test_noise.tolist()}, f"models/wgan_gp/{settings.MODEL_NAME}/metadata.pkl")
    exit(-1)

  save_model(gen, optimizer_gen, f"models/wgan_gp/{settings.MODEL_NAME}/gen.mod")
  save_model(crit, optimizer_crit, f"models/wgan_gp/{settings.MODEL_NAME}/crit.mod")
  save_metadata({"iteration": iteration, "noise": test_noise.tolist()}, f"models/wgan_gp/{settings.MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
    train()