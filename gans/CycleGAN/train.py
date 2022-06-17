import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import os
import pathlib

import settings
from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata
from gans.utils.datasets import SplitImagePairDataset
from generator_model import Generator
from discriminator_model import Discriminator

def train(imageA, imageB, disc_A, disc_B, gen_A, gen_B, opt_disc, opt_gen, L1_loss, gan_loss, d_scaler, g_scaler):
  imageA = imageA.to(settings.device)
  imageB = imageB.to(settings.device)

  # Train discriminators
  with torch.cuda.amp.autocast():
    # Discriminator A
    fake_A = gen_A(imageB)
    D_A_real = disc_A(imageA)
    D_A_fake = disc_A(fake_A.detach())

    D_A_real_loss = gan_loss(D_A_real, (torch.ones_like(D_A_real) - 0.1 * torch.rand_like(D_A_real)) if settings.TRUE_LABEL_SMOOTHING else torch.ones_like(D_A_real))
    D_A_fake_loss = gan_loss(D_A_fake, (torch.zeros_like(D_A_fake) + 0.1 * torch.rand_like(D_A_fake)) if settings.FAKE_LABEL_SMOOTHING else torch.zeros_like(D_A_fake))
    D_A_loss = D_A_real_loss + D_A_fake_loss

    # Discriminator B
    fake_B = gen_B(imageA)
    D_B_real = disc_B(imageB)
    D_B_fake = disc_B(fake_B.detach())

    D_B_real_loss = gan_loss(D_B_real, (torch.ones_like(D_B_real) - 0.1 * torch.rand_like(D_B_real)) if settings.TRUE_LABEL_SMOOTHING else torch.ones_like(D_B_real))
    D_B_fake_loss = gan_loss(D_B_fake, (torch.zeros_like(D_B_fake) + 0.1 * torch.rand_like(D_B_fake)) if settings.FAKE_LABEL_SMOOTHING else torch.zeros_like(D_B_fake))
    D_B_loss = D_B_real_loss + D_B_fake_loss

    # Together
    D_loss = (D_A_loss + D_B_loss) / 2

  opt_disc.zero_grad()
  d_scaler.scale(D_loss).backward()
  d_scaler.step(opt_disc)
  d_scaler.update()

  # Train generators
  with torch.cuda.amp.autocast():
    # Adverserial loss
    D_A_fake = disc_A(fake_A)
    D_B_fake = disc_B(fake_B)

    G_A_loss = gan_loss(D_A_fake, torch.ones_like(D_A_fake))
    G_B_loss = gan_loss(D_B_fake, torch.ones_like(D_B_fake))

    # Cycle loss
    cycle_A = gen_A(fake_B)
    cycle_B = gen_B(fake_A)

    cycle_A_loss = L1_loss(imageA, cycle_A)
    cycle_B_loss = L1_loss(imageB, cycle_B)

    # Together
    G_loss = G_A_loss + G_B_loss + cycle_A_loss * settings.LAMBDA_CYCLE + cycle_B_loss * settings.LAMBDA_CYCLE

    if settings.LAMBDA_IDENTITY > 0:
      # Identity loss only when its lambda is larger than zero
      identity_A = gen_A(imageA)
      identity_B = gen_B(imageB)

      identity_A_loss = L1_loss(imageA, identity_A)
      identity_B_loss = L1_loss(imageB, identity_B)

      G_loss += identity_A_loss * settings.LAMBDA_IDENTITY + identity_B_loss * settings.LAMBDA_IDENTITY

  opt_gen.zero_grad()
  g_scaler.scale(G_loss).backward()
  g_scaler.step(opt_gen)
  g_scaler.update()

  return D_loss, G_loss


def main():
  disc_A = Discriminator(channels=settings.IMG_CHAN, features=settings.FEATURES_DISC).to(settings.device)
  disc_B = Discriminator(channels=settings.IMG_CHAN, features=settings.FEATURES_DISC).to(settings.device)
  gen_A = Generator(in_channels=settings.IMG_CHAN, features=settings.FEATURES_GEN, residuals=settings.GEN_RESIDUAL).to(settings.device)
  gen_B = Generator(in_channels=settings.IMG_CHAN, features=settings.FEATURES_GEN, residuals=settings.GEN_RESIDUAL).to(settings.device)

  opt_disc = optim.Adam(list(disc_A.parameters()) + list(disc_B.parameters()), lr=settings.LR, betas=(0.5, 0.999))
  opt_gen = optim.Adam(list(gen_A.parameters()) + list(gen_B.parameters()), lr=settings.LR, betas=(0.5, 0.999))

  L1_loss = nn.L1Loss()
  gan_loss = nn.MSELoss()

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/genA.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/genA.mod"):
      load_model(f"models/{settings.MODEL_NAME}/genA.mod", gen_A, opt_gen, settings.LR, settings.device)
  except:
    print("Generator model A is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/discA.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/discA.mod"):
      load_model(f"models/{settings.MODEL_NAME}/discA.mod", disc_A, opt_disc, settings.LR, settings.device)
  except:
    print("Discriminator model A is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/genB.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/genB.mod"):
      load_model(f"models/{settings.MODEL_NAME}/genB.mod", gen_B, opt_gen, settings.LR, settings.device)
  except:
    print("Generator model B is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/discB.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/discB.mod"):
      load_model(f"models/{settings.MODEL_NAME}/discB.mod", disc_B, opt_disc, settings.LR, settings.device)
  except:
    print("Discriminator model B is incompatible with found model parameters")

  metadata = None
  if os.path.exists(f"models/{settings.MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{settings.MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{settings.MODEL_NAME}/metadata.pkl")

  iteration = 0
  if metadata is not None:
    if "iteration" in metadata.keys():
      iteration = int(metadata["iteration"])

  if settings.GEN_A_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.GEN_A_MODEL_WEIGHTS_TO_LOAD, gen_A, opt_gen, settings.LR, settings.device)
    except Exception as e:
      print(f"Generator model A weights are incompatible with found model parameters\n{e}")
      exit(2)

  if settings.DISC_A_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.DISC_A_MODEL_WEIGHTS_TO_LOAD, disc_A, opt_disc, settings.LR, settings.device)
    except Exception as e:
      print(f"Discriminator model A weights are incompatible with found model parameters\n{e}")
      exit(2)

  if settings.GEN_B_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.GEN_B_MODEL_WEIGHTS_TO_LOAD, gen_B, opt_gen, settings.LR, settings.device)
    except Exception as e:
      print(f"Generator model B weights are incompatible with found model parameters\n{e}")
      exit(2)

  if settings.DISC_B_MODEL_WEIGHTS_TO_LOAD is not None:
    try:
      load_model(settings.DISC_B_MODEL_WEIGHTS_TO_LOAD, disc_B, opt_disc, settings.LR, settings.device)
    except Exception as e:
      print(f"Discriminator model B weights are incompatible with found model parameters\n{e}")
      exit(2)

  dataset_format = "RGB" if settings.IMG_CHAN == 3 else "GRAY"
  train_dataset = SplitImagePairDataset(root=settings.TRAIN_DIR, transform=settings.transforms, format=dataset_format)
  train_dataloader = DataLoader(train_dataset, settings.BATCH_SIZE, True, num_workers=settings.WORKERS, persistent_workers=True, pin_memory=True)
  dataset_length = len(train_dataset)
  number_of_batches = dataset_length // settings.BATCH_SIZE
  number_of_epochs = settings.ITERATIONS // number_of_batches
  print(f"Found {dataset_length} training images in {number_of_batches} batches")
  print(f"Which corespondes to {number_of_epochs} epochs")
  test_dataset = SplitImagePairDataset(root=settings.VAL_DIR, transform=settings.test_transform, format=dataset_format)
  test_dataloader = DataLoader(test_dataset, settings.TESTING_SAMPLES, False, pin_memory=True)

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    with tqdm(total=settings.ITERATIONS, initial=iteration, unit="it") as bar:
      while True:
        for x, y in train_dataloader:
          d_loss, g_loss = train(x, y, disc_A, disc_B, gen_A, gen_B, opt_disc, opt_gen, L1_loss, gan_loss, d_scaler, g_scaler)

          if d_loss is not None and g_loss is not None:
            summary_writer.add_scalar("Gen Loss", g_loss, global_step=iteration)
            summary_writer.add_scalar("Disc Loss", d_loss, global_step=iteration)

          if iteration % settings.CHECKPOINT_EVERY == 0 and settings.SAVE_CHECKPOINTS:
            save_model(gen_A, opt_gen, f"models/{settings.MODEL_NAME}/{iteration}_genA.mod")
            save_model(disc_A, opt_disc, f"models/{settings.MODEL_NAME}/{iteration}_discA.mod")
            save_model(gen_B, opt_gen, f"models/{settings.MODEL_NAME}/{iteration}_genB.mod")
            save_model(disc_B, opt_disc, f"models/{settings.MODEL_NAME}/{iteration}_discB.mod")

          if iteration % settings.SAMPLE_EVERY == 0:
            print(f"Loss disc: {d_loss:.4f}, Loss gen: {g_loss:.4f}")

            imgA, imgB = next(iter(test_dataloader))
            imgA, imgB = imgA.to(settings.device), imgB.to(settings.device)
            gen_A.eval()
            gen_B.eval()

            with torch.no_grad():
              fake_A = gen_A(imgB)
              fake_B = gen_B(imgA)

              fake_AToB = gen_B(fake_A)
              fake_BToA = gen_A(fake_B)

              img_AtoB = torchvision.utils.make_grid(fake_B[:settings.TESTING_SAMPLES], normalize=True)
              img_BtoA = torchvision.utils.make_grid(fake_A[:settings.TESTING_SAMPLES], normalize=True)
              img_AtoBtoA = torchvision.utils.make_grid(fake_BToA[:settings.TESTING_SAMPLES], normalize=True)
              img_BtoAtoB = torchvision.utils.make_grid(fake_AToB[:settings.TESTING_SAMPLES], normalize=True)

              if iteration == 0:
                imgA_grid = torchvision.utils.make_grid(imgA[:settings.TESTING_SAMPLES], normalize=True)
                imgB_grid = torchvision.utils.make_grid(imgB[:settings.TESTING_SAMPLES], normalize=True)
                summary_writer.add_image("A", imgA_grid, global_step=iteration)
                summary_writer.add_image("B", imgB_grid, global_step=iteration)

              summary_writer.add_image("A to B", img_AtoB, global_step=iteration)
              summary_writer.add_image("B to A", img_BtoA, global_step=iteration)
              summary_writer.add_image("A to B to A", img_AtoBtoA, global_step=iteration)
              summary_writer.add_image("B to A to B", img_BtoAtoB, global_step=iteration)

            gen_A.train()
            gen_B.train()

          bar.update()
          iteration += 1
          if iteration > settings.ITERATIONS:
            break

        if iteration > settings.ITERATIONS:
          break

        save_model(gen_A, opt_gen, f"models/{settings.MODEL_NAME}/genA.mod")
        save_model(disc_A, opt_disc, f"models/{settings.MODEL_NAME}/discA.mod")
        save_model(gen_B, opt_gen, f"models/{settings.MODEL_NAME}/genB.mod")
        save_model(disc_B, opt_disc, f"models/{settings.MODEL_NAME}/discB.mod")
        save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
  except KeyboardInterrupt:
    print("Exiting")
  except Exception as e:
    print(e)
    save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
    exit(-1)

  save_model(gen_A, opt_gen, f"models/{settings.MODEL_NAME}/genA.mod")
  save_model(disc_A, opt_disc, f"models/{settings.MODEL_NAME}/discA.mod")
  save_model(gen_B, opt_gen, f"models/{settings.MODEL_NAME}/genB.mod")
  save_model(disc_B, opt_disc, f"models/{settings.MODEL_NAME}/discB.mod")
  save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
  main()
