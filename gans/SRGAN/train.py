import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

import settings
from model import Discriminator, Generator
from dataset import SrganDataset
from utils import VGGLoss, gradient_penalty
from gans.utils.training_saver import save_model, load_model, load_metadata, save_metadata

torch.backends.cudnn.benchmark = True

def train(opt_disc, opt_gen, disc, gen, train_dataset_loader, mse, bce, vgg_loss, d_scaler, g_scaler, pretrain):
  loop = tqdm(train_dataset_loader, leave=True, unit="batch")

  D_loss, G_loss = None, None
  for idx, (low_res, high_res) in enumerate(loop):
    low_res = low_res.to(settings.device)
    high_res = high_res.to(settings.device)

    if pretrain:
      with torch.cuda.amp.autocast():
        fake = gen(low_res)
        G_loss = mse(fake, high_res)

      opt_gen.zero_grad()
      g_scaler.scale(G_loss).backward()
      g_scaler.step(opt_gen)
      g_scaler.update()
    else:
      # Train discriminator
      with torch.cuda.amp.autocast():
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())

        disc_loss_real = bce(disc_real, (torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)) if settings.TRUE_LABEL_SMOOTHING else torch.ones_like(disc_real))
        disc_loss_fake = bce(disc_fake, (torch.zeros_like(disc_fake) + 0.1 * torch.rand_like(disc_fake)) if settings.FAKE_LABEL_SMOOTHING else torch.zeros_like(disc_fake))
        D_loss = disc_loss_fake + disc_loss_real

        if settings.GP_LAMBDA != 0:
          D_loss += settings.GP_LAMBDA * gradient_penalty(disc, high_res, fake, settings.device)

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

  return D_loss, G_loss

if __name__ == '__main__':
  disc = Discriminator(settings.IMG_CH, settings.DISC_FEATURES).to(settings.device)
  gen = Generator(settings.IMG_CH, settings.GEN_FEATURES).to(settings.device)

  opt_gen = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0.9 if settings.GP_LAMBDA == 0 else 0.5, 0.999 if settings.GP_LAMBDA == 0 else 0.9))
  opt_disc = optim.Adam(disc.parameters(), lr=settings.LR, betas=(0.9 if settings.GP_LAMBDA == 0 else 0.5, 0.999 if settings.GP_LAMBDA == 0 else 0.9))

  mse = nn.MSELoss()
  bce = nn.BCEWithLogitsLoss()
  vgg_loss = VGGLoss()

  train_dataset = SrganDataset(root_dir=settings.DATASET_PATH)
  train_dataset_loader = DataLoader(
    train_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=settings.NUM_OF_WORKERS,
    persistent_workers=True
  )

  test_dataset = SrganDataset(root_dir=settings.TEST_DATASET_PATH, transforms=[settings.high_res_test_transform, settings.low_res_test_transform])
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

  start_epoch = 0
  if os.path.exists(f"models/{settings.MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{settings.MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{settings.MODEL_NAME}/metadata.pkl")

    if "epoch" in metadata.keys():
      start_epoch = int(metadata["epoch"])

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  last_epoch = 0
  try:
    for epoch in range(start_epoch, settings.EPOCHS + settings.PRETRAIN_EPOCHS):
      last_epoch = epoch

      pretrain = epoch < settings.PRETRAIN_EPOCHS
      disc_loss, gen_loss = train(opt_disc, opt_gen, disc, gen, train_dataset_loader, mse, bce, vgg_loss, d_scaler, g_scaler, pretrain)

      stats = []
      if gen_loss is not None:
        stats.append(f"Loss gen: {gen_loss:.4f}")
        summary_writer.add_scalar("Gen Loss", gen_loss, global_step=epoch)
      if disc_loss is not None:
        stats.append(f"Loss disc: {disc_loss:.4f}")
        summary_writer.add_scalar("Disc Loss", disc_loss, global_step=epoch)

      print(f"Epoch: {epoch}/{settings.EPOCHS + settings.PRETRAIN_EPOCHS} {', '.join(stats)}")

      if settings.SAVE_CHECKPOINTS and epoch % settings.CHECKPOINT_EVERY == 0:
        save_model(gen, opt_gen, f"models/{settings.MODEL_NAME}/gen_{epoch}.mod")
        if not pretrain:
          save_model(disc, opt_disc, f"models/{settings.MODEL_NAME}/disc_{epoch}.mod")

      save_model(gen, opt_gen, f"models/{settings.MODEL_NAME}/gen.mod")
      if not pretrain:
        save_model(disc, opt_disc, f"models/{settings.MODEL_NAME}/disc.mod")
      save_metadata({"epoch": last_epoch}, f"models/{settings.MODEL_NAME}/metadata.pkl")

      if epoch % settings.SAMPLE_EVERY == 0:
        test_images = next(iter(test_dataset_loader))
        input_imgs = test_images[1]
        true_imgs = test_images[0]
        input_imgs = input_imgs.to(settings.device)

        gen.eval()

        with torch.no_grad():
          upscaled_images = gen(input_imgs)

          input_images_grid = torchvision.utils.make_grid(input_imgs[:settings.TESTING_SAMPLES], normalize=True)
          upscaled_images_grid = torchvision.utils.make_grid(upscaled_images[:settings.TESTING_SAMPLES], normalize=True)
          original_images_grid = torchvision.utils.make_grid(true_imgs[:settings.TESTING_SAMPLES], normalize=True)

          summary_writer.add_image("Original", original_images_grid, global_step=epoch)
          summary_writer.add_image("Input", input_images_grid, global_step=epoch)
          summary_writer.add_image("Upscaled", upscaled_images_grid, global_step=epoch)

        gen.train()

  except KeyboardInterrupt:
    print("Exiting")
    pass

  save_model(gen, opt_gen, f"models/{settings.MODEL_NAME}/gen.mod")
  save_model(disc, opt_disc, f"models/{settings.MODEL_NAME}/disc.mod")
  save_metadata({"epoch": last_epoch}, f"models/{settings.MODEL_NAME}/metadata.pkl")
