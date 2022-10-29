import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pathlib
from tqdm import tqdm
import traceback

from gans.VQGAN import settings
from gans.VQGAN.reconstructor import Reconstructor
from gans.VQGAN.discriminator import Discriminator
from gans.VQGAN.lpips import LPIPS
from gans.utils.helpers import initialize_model
from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata
from gans.utils.datasets import SingleInSingleOutDataset

torch.backends.cudnn.benchmark = True

def train_step(images, disc, rec, optimizer_disc, optimizer_rec, perceptual_loss, iteration):
  decoded_images, _, q_loss = rec(images)

  disc_real_loss = disc(images)
  disc_fake_loss = disc(decoded_images)

  perceptual_loss = perceptual_loss(images, decoded_images)
  rec_loss = torch.abs(images - decoded_images)
  perceptual_rec_loss = settings.PERCEPTUAL_LOSS_FACTOR * perceptual_loss + settings.RECONSTRUCTION_LOSS_FACTOR * rec_loss
  perceptual_rec_loss = perceptual_rec_loss.mean()
  g_loss = -torch.mean(disc_fake_loss)

  l = rec.calculate_lambda(perceptual_rec_loss, g_loss)
  rec_loss = perceptual_rec_loss + q_loss
  if iteration > settings.DISC_TRAINING_DELAY_ITERATIONS:
    rec_loss += settings.DISC_FACTOR * l * g_loss

  optimizer_rec.zero_grad()
  rec_loss.backward(retain_graph=True)

  disc_loss = None
  if iteration > settings.DISC_TRAINING_DELAY_ITERATIONS:
    d_loss_real = torch.mean(F.relu(1. - disc_real_loss))
    d_loss_fake = torch.mean(F.relu(1. + disc_fake_loss))
    disc_loss = settings.DISC_FACTOR * 0.5 * (d_loss_real + d_loss_fake)

    optimizer_disc.zero_grad()
    disc_loss.backward()

  optimizer_rec.step()
  optimizer_disc.step()

  return disc_loss.item() if disc_loss is not None else 0.0, rec_loss.item()

def main():
  reconstructor = Reconstructor(settings.ENCODER_FILTERS, settings.DECODER_FILTERS,
                                settings.ATTENTION_RESOLUTIONS,
                                settings.ENCODER_RESIDUAL_PER_LEVEL, settings.DECODER_RESIDUAL_PER_LEVEL,
                                settings.IMG_SIZE, settings.IMG_CH,
                                settings.LATENT_DIMENSION, settings.NUMBER_OF_LATENT_VECTORS,
                                settings.BETA,
                                settings.device).to(settings.device)
  discriminator = Discriminator(settings.DISC_BASE_FILTERS, settings.DISC_MAX_FILTER_MULTIPLIER, settings.DISC_LAYERS,
                                settings.IMG_CH).to(settings.device)
  initialize_model(discriminator)

  perceptual_loss = LPIPS().eval().to(settings.device)

  rec_opt = optim.Adam(
    list(reconstructor.encoder.parameters()) +
    list(reconstructor.decoder.parameters()) +
    list(reconstructor.codebook.parameters()) +
    list(reconstructor.quant_conv.parameters()) +
    list(reconstructor.post_quant_conv.parameters()),
    lr=settings.LR, eps=1e-08, betas=(settings.OPT_BETA1, settings.OPT_BETA2)
  )
  disc_opt = optim.Adam(discriminator.parameters(),
                        lr=settings.LR, eps=1e-08, betas=(settings.OPT_BETA1, settings.OPT_BETA2))

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/rec.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/rec.mod"):
      load_model(f"models/{settings.MODEL_NAME}/rec.mod", reconstructor, rec_opt, settings.LR, settings.device)
  except:
    print("Reconstructor model is incompatible with found model parameters")

  try:
    if os.path.exists(f"models/{settings.MODEL_NAME}/disc.mod") and os.path.isfile(f"models/{settings.MODEL_NAME}/disc.mod"):
      load_model(f"models/{settings.MODEL_NAME}/disc.mod", discriminator, disc_opt, settings.LR, settings.device)
  except:
    print("Discriminator model is incompatible with found model parameters")

  metadata = None
  if os.path.exists(f"models/{settings.MODEL_NAME}/metadata.pkl") and os.path.isfile(f"models/{settings.MODEL_NAME}/metadata.pkl"):
    metadata = load_metadata(f"models/{settings.MODEL_NAME}/metadata.pkl")

  iteration = 0
  if metadata is not None:
    if "iteration" in metadata.keys():
      iteration = int(metadata["iteration"])

  data_format = "RGB" if settings.IMG_CH == 3 else "GRAY"
  dataset = SingleInSingleOutDataset(root_dir=settings.DATASET_PATH, transform=settings.training_transform, format=data_format)
  loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_OF_WORKERS, persistent_workers=True, pin_memory=True)
  val_dataset = SingleInSingleOutDataset(root_dir=settings.VAL_DATASET_PATH, transform=settings.eval_transform, format=data_format)
  val_loader = DataLoader(val_dataset, batch_size=settings.NUMBER_OF_SAMPLES, shuffle=False, num_workers=settings.NUM_OF_WORKERS, persistent_workers=True, pin_memory=True)
  steps_per_epoch = len(dataset) // settings.BATCH_SIZE
  steps_per_training = settings.EPOCHS * steps_per_epoch

  if not os.path.exists(f"models/{settings.MODEL_NAME}"):
    pathlib.Path(f"models/{settings.MODEL_NAME}").mkdir(parents=True, exist_ok=True)

  try:
    with tqdm(total=steps_per_training, initial=iteration, unit="it") as bar:
      while True:
        for images in loader:
          images = images.to(settings.device)
          disc_loss, gen_loss = train_step(images, discriminator, reconstructor, disc_opt, rec_opt, perceptual_loss, iteration)
          summary_writer.add_scalar("Rec Loss", gen_loss, global_step=iteration)
          summary_writer.add_scalar("Disc Loss", disc_loss, global_step=iteration)

          if iteration % settings.CHECKPOINT_EVERY == 0:
            save_model(reconstructor, rec_opt, f"models/{settings.MODEL_NAME}/{iteration}_rec.mod")
            save_model(discriminator, disc_opt, f"models/{settings.MODEL_NAME}/{iteration}_disc.mod")
            save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")

          if iteration % settings.SAMPLE_EVERY == 0:
            test_images = next(iter(val_loader)).to(settings.device)

            reconstructor.eval()

            with torch.no_grad():
              decoded_images, _, _ = reconstructor(test_images)

            if iteration == 0:
              reference_images = torchvision.utils.make_grid(test_images[:settings.NUMBER_OF_SAMPLES], normalize=True)
              summary_writer.add_image("Reconstruction reference", reference_images, global_step=iteration)

            reconstructed_images = torchvision.utils.make_grid(decoded_images[:settings.NUMBER_OF_SAMPLES], normalize=True)
            summary_writer.add_image("Reconstructed", reconstructed_images, global_step=iteration)

            reconstructor.train()

          bar.update()
          iteration += 1

          bar.set_description(f"Loss disc: {disc_loss:.4f}, Loss rec: {gen_loss:.4f}", refresh=False)

          if iteration > steps_per_training:
            break

        if iteration > steps_per_training:
          break
  except KeyboardInterrupt:
    print("Exiting")
  except Exception as e:
    print(traceback.format_exc())
    save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")
    exit(-1)

  save_model(reconstructor, rec_opt, f"models/{settings.MODEL_NAME}/rec.mod")
  save_model(discriminator, disc_opt, f"models/{settings.MODEL_NAME}/disc.mod")
  save_metadata({"iteration": iteration}, f"models/{settings.MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
  main()
