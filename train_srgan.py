import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import traceback

import gans.SRGAN.settings as settings
from gans.SRGAN.model import Discriminator, Generator
from gans.SRGAN.utils import VGGLoss, gradient_penalty
from gans.utils.training_saver import save_model, load_model, load_metadata, save_metadata
from gans.utils.datasets import SingleInTwoOutDataset
from gans.utils.learning import get_linear_lr_decay_scheduler

torch.backends.cudnn.benchmark = True

def train(high_res, low_res, opt_disc, opt_gen, disc, gen, bce, mse, vgg_loss, d_scaler, g_scaler, pretrain):
  low_res = low_res.to(settings.device)
  high_res = high_res.to(settings.device)

  # Train discriminator
  D_loss = None
  if not pretrain:
    with torch.cuda.amp.autocast():
      fake = gen(low_res)
      disc_real = disc(high_res)
      disc_fake = disc(fake.detach())

      disc_loss_real = bce(disc_real, (torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)) if settings.TRUE_LABEL_SMOOTHING else torch.ones_like(disc_real))
      disc_loss_fake = bce(disc_fake, (torch.zeros_like(disc_fake) + 0.1 * torch.rand_like(disc_fake)) if settings.FAKE_LABEL_SMOOTHING else torch.zeros_like(disc_fake))
      D_loss = disc_loss_fake + disc_loss_real

      if settings.GP_LAMBDA > 0:
        gp = gradient_penalty(disc, high_res, fake, device=settings.device)
        D_loss += settings.GP_LAMBDA * gp

    opt_disc.zero_grad()
    d_scaler.scale(D_loss).backward()
    d_scaler.step(opt_disc)
    d_scaler.update()

    # Train generator
    with torch.cuda.amp.autocast():
      disc_fake = disc(fake)

      adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
      loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
      G_loss = loss_for_vgg + adversarial_loss

    opt_gen.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(opt_gen)
    g_scaler.update()
  else:
    # Pretrain generator
    with torch.cuda.amp.autocast():
      fake = gen(low_res)
      G_loss = mse(fake, high_res)

    opt_gen.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(opt_gen)
    g_scaler.update()

  return D_loss.item() if D_loss is not None else 0, G_loss

if __name__ == '__main__':
  disc = Discriminator(settings.IMG_CH, settings.DISC_FEATURES).to(settings.device)
  gen = Generator(settings.IMG_CH, settings.GEN_FEATURES).to(settings.device)

  opt_gen = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0.9 if settings.GP_LAMBDA == 0 else 0.5, 0.999 if settings.GP_LAMBDA == 0 else 0.9))
  opt_disc = optim.Adam(disc.parameters(), lr=settings.LR, betas=(0.9 if settings.GP_LAMBDA == 0 else 0.5, 0.999 if settings.GP_LAMBDA == 0 else 0.9))

  mse = nn.MSELoss()
  bce = nn.BCEWithLogitsLoss()
  vgg_loss = VGGLoss()

  dataset_format = "RGB" if settings.IMG_CH == 3 else "GRAY"
  train_dataset = SingleInTwoOutDataset(root_dir=settings.DATASET_PATH, both_transform=settings.both_transform, first_transform=settings.high_res_transform, second_transform=settings.low_res_transform, format=dataset_format)
  train_dataset_loader = DataLoader(
    train_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=settings.NUM_OF_WORKERS,
    persistent_workers=True
  )

  test_dataset = SingleInTwoOutDataset(root_dir=settings.TEST_DATASET_PATH, both_transform=settings.both_test_transform, first_transform=settings.high_res_test_transform, second_transform=settings.low_res_test_transform, format=dataset_format)
  test_dataset_loader = DataLoader(
    test_dataset,
    batch_size=settings.TESTING_SAMPLES,
    pin_memory=True
  )

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/disc.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/disc.mod"):
      load_model(f"models/{settings.MODEL_NAME}/disc.mod", disc, opt_disc, settings.LR, settings.device)
  except:
    print("Discriminator model is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/gen.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/gen.mod"):
      load_model(f"models/{settings.MODEL_NAME}/gen.mod", gen, opt_gen, settings.LR, settings.device)
  except:
    print("Generator model is incompatible with found model parameters")

  if settings.GEN_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.GEN_MODEL_WEIGHTS_TO_LOAD, gen, opt_gen, settings.LR, settings.device)
    except Exception as e:
      print(f"Generator model weights are incompatible with found model parameters\n{e}")
      exit(2)

  if settings.DISC_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.DISC_MODEL_WEIGHTS_TO_LOAD, disc, opt_disc, settings.LR, settings.device)
    except Exception as e:
      print(f"Discriminator model weights are incompatible with found model parameters\n{e}")
      exit(2)

  iteration = 0
  if os.path.exists(f"models/{settings.MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{settings.MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{settings.MODEL_NAME}/metadata.pkl")

    if "iteration" in metadata.keys():
      iteration = int(metadata["iteration"])

  schedulers = []
  if settings.DECAY_LR:
    schedulers.append(get_linear_lr_decay_scheduler(opt_gen, settings.DECAY_AFTER_ITERATIONS, settings.DECAY_ITERATION, iteration + 1))
    schedulers.append(get_linear_lr_decay_scheduler(opt_disc, settings.DECAY_AFTER_ITERATIONS, settings.DECAY_ITERATION, iteration + 1))

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  total_iterations = settings.ITERATIONS + settings.PRETRAIN_ITERATIONS
  pretrain = iteration < settings.PRETRAIN_ITERATIONS

  try:
    with tqdm(total=total_iterations, initial=iteration, unit="it") as bar:
      while True:
        for high_res, low_res in train_dataset_loader:
          pretrain = iteration < settings.PRETRAIN_ITERATIONS
          disc_loss, gen_loss = train(high_res, low_res, opt_disc, opt_gen, disc, gen, bce, mse, vgg_loss, d_scaler, g_scaler, pretrain)

          summary_writer.add_scalar("Gen Loss", gen_loss, global_step=iteration)
          summary_writer.add_scalar("Disc Loss", disc_loss, global_step=iteration)
          bar.set_description(f"Loss disc: {disc_loss:.4f}, Loss gen: {gen_loss:.4f}, Lr: {opt_gen.param_groups[0]['lr']:.7f}", refresh=False)

          if settings.SAVE_CHECKPOINTS and iteration % settings.CHECKPOINT_EVERY == 0:
            save_model(gen, opt_gen, f"models/{settings.MODEL_NAME}/{iteration}_gen.mod")
            if not pretrain:
              save_model(disc, opt_disc, f"models/{settings.MODEL_NAME}/{iteration}_disc.mod")

          if iteration % settings.SAMPLE_EVERY == 0:
            true_imgs, input_imgs = next(iter(test_dataset_loader))
            input_imgs = input_imgs.to(settings.device)

            gen.eval()

            with torch.no_grad():
              upscaled_images = gen(input_imgs)
              upscaled_images_grid = torchvision.utils.make_grid(upscaled_images[:settings.TESTING_SAMPLES], normalize=True)

              summary_writer.add_image("Upscaled", upscaled_images_grid, global_step=iteration)

              if iteration == 0:
                original_images_grid = torchvision.utils.make_grid(true_imgs[:settings.TESTING_SAMPLES], normalize=True)
                input_images_grid = torchvision.utils.make_grid(input_imgs[:settings.TESTING_SAMPLES], normalize=True)

                summary_writer.add_image("Original", original_images_grid, global_step=iteration)
                summary_writer.add_image("Input", input_images_grid, global_step=iteration)

            gen.train()

          bar.update()
          iteration += 1

          for scheduler in schedulers:
            scheduler.step()

          if iteration > total_iterations:
            break

        if iteration > total_iterations:
          break

        save_model(gen, opt_gen, f"models/{settings.MODEL_NAME}/gen.mod")
        if not pretrain:
          save_model(disc, opt_disc, f"models/{settings.MODEL_NAME}/disc.mod")
        save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")
  except Exception as e:
    print(traceback.format_exc())
    save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
    exit(-1)

  save_model(gen, opt_gen, f"models/{settings.MODEL_NAME}/gen.mod")
  if not pretrain:
    save_model(disc, opt_disc, f"models/{settings.MODEL_NAME}/disc.mod")
  save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
