import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import pathlib

from gans.utils.training_saver import load_model, save_model, save_metadata, load_metadata
from gans.utils.helpers import inception_score
from gans.ProGAN.model import Critic, Generator
import gans.ProGAN.settings as settings

from gans.utils.datasets import SingleInSingleOutDataset

torch.backends.cudnn.benchmark = True

def gradient_penalty(critic, real, fake, alpha, step, device):
  batch, channels, height, width = real.shape
  epsilon = torch.rand((batch, 1, 1, 1)).repeat(1, channels, height, width).to(device)
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

  assert image_size in settings.IMG_SIZE_TO_BATCH_SIZE.keys(), f"Missing batch size for image of size {image_size} in settings"

  batch_size = settings.IMG_SIZE_TO_BATCH_SIZE[image_size]
  dataset = SingleInSingleOutDataset(root_dir=settings.DATASET_PATH, transform=transform, format="RGB" if settings.IMG_CH == 3 else "GRAY")
  loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=settings.NUM_OF_WORKERS, pin_memory=True, persistent_workers=True)
  return loader, dataset

def train_step(data, crit, gen, stage, alpha, opt_critic, opt_generator, c_scaler, g_scaler, batch_repeats):
  real = data.to(settings.device)
  cur_batch_size = real.shape[0]

  opt_critic.zero_grad()
  for _ in range(batch_repeats):
    # Train critic
    noise = torch.randn((cur_batch_size, settings.Z_DIM, 1, 1), device=settings.device)

    with torch.cuda.amp.autocast():
      fake = gen(noise, alpha, stage)
      critic_real = crit(real, alpha, stage)
      critic_fake = crit(fake.detach(), alpha, stage)

      loss_crit = -(torch.mean(critic_real) - torch.mean(critic_fake))
      if settings.LAMBDA_GP != 0:
        gp = gradient_penalty(crit, real, fake, alpha, stage, device=settings.device)
        loss_crit += settings.LAMBDA_GP * gp
      if settings.EPSILON_DRIFT != 0:
        loss_crit += settings.EPSILON_DRIFT * torch.mean(critic_real ** 2)

    c_scaler.scale(loss_crit).backward()

  c_scaler.step(opt_critic)
  c_scaler.update()

  opt_generator.zero_grad()
  for _ in range(batch_repeats):
    # Train generator
    noise = torch.randn((cur_batch_size, settings.Z_DIM, 1, 1), device=settings.device)

    with torch.cuda.amp.autocast():
      fake = gen(noise, alpha, stage)
      gen_fake = crit(fake, alpha, stage)
      loss_gen = -torch.mean(gen_fake)

    g_scaler.scale(loss_gen).backward()

  g_scaler.step(opt_generator)
  g_scaler.update()

  return loss_crit, loss_gen

global_step = 0
iteration = 0
def train_loop(fade_iterations, train_iterations, train_stage, alpha, crit, gen, opt_critic, opt_generator, c_scaler, g_scaler, summary_writer, test_noise, batch_repeats):
  global global_step
  global iteration

  number_of_iterations = fade_iterations + train_iterations
  img_size = 4 * 2 ** train_stage
  loader, dataset = get_loader(img_size)
  number_of_batches = len(loader)
  number_of_epochs = number_of_iterations / number_of_batches

  print(f"Starting image size: {img_size} with batch size: {settings.IMG_SIZE_TO_BATCH_SIZE[img_size]} which coresponds to {number_of_batches} number of batches\nTraining for {number_of_epochs:.2f} epochs ({int(number_of_iterations * settings.IMG_SIZE_TO_BATCH_SIZE[img_size])} images)")
  print(f"Fading iterations: {fade_iterations} ({fade_iterations * settings.IMG_SIZE_TO_BATCH_SIZE[img_size]} images), Full train iterations: {train_iterations} ({train_iterations * settings.IMG_SIZE_TO_BATCH_SIZE[img_size]} images)")
  with tqdm(total=number_of_iterations, initial=iteration, unit="it") as bar:
    while True:
      for data in loader:
        crit_loss, gen_loss = train_step(data, crit, gen, train_stage, alpha, opt_critic, opt_generator, c_scaler, g_scaler, batch_repeats)
        alpha = max(settings.START_ALPHA, (1.0 * (iteration / fade_iterations)) if fade_iterations != 0 else 1.0)
        alpha = min(alpha, 1.0)

        if gen_loss is not None:
          summary_writer.add_scalar("Gen Loss", gen_loss, global_step=global_step)
        if crit_loss is not None:
          summary_writer.add_scalar("Crit Loss", crit_loss, global_step=global_step)
        summary_writer.add_scalar("Alpha", alpha, global_step=global_step)

        if settings.SAVE_CHECKPOINTS and iteration % settings.CHECKPOINT_EVERY == 0:
          save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen_{global_step}.mod")
          save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit_{global_step}.mod")

        if iteration % settings.SAMPLE_EVERY == 0:
          gen.eval()

          with torch.no_grad():
            fake = gen(test_noise, alpha, train_stage)
            img_grid_fake = torchvision.utils.make_grid(fake[:settings.TESTING_SAMPLES], normalize=True)
            summary_writer.add_image("Generated", img_grid_fake, global_step=global_step)

            if alpha >= 1.0:
              inception_image_batches = []
              for _ in range(settings.INCEPTION_SCORE_NUMBER_OF_BATCHES):
                inception_noise = torch.randn((settings.INCEPTION_SCORE_BATCH_SIZE, settings.Z_DIM, 1, 1), device=settings.device)
                inception_image_batches.append(gen(inception_noise, alpha, train_stage))

              mean_inception_score, inception_score_stddev = inception_score(inception_image_batches, settings.INCEPTION_SCORE_BATCH_SIZE, True, splits=settings.INCEPTION_SCORE_SPLIT)
              print(f"\nLoss crit: {crit_loss:.4f}, Loss gen: {gen_loss:.4f}, Inception Score: {mean_inception_score:.2f}±{inception_score_stddev:.2f}")
              summary_writer.add_scalar("Inception Score", mean_inception_score, global_step=global_step)
            else:
              print(f"\nLoss crit: {crit_loss:.4f}, Loss gen: {gen_loss:.4f}, Alpha: {alpha:.4f}")

          gen.train_step()

        iteration += 1

        if iteration > number_of_iterations:
          break

        bar.update()
        global_step += 1
      if iteration > number_of_iterations:
        break

      save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
      save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit.mod")
      save_metadata({"iteration": iteration, "train_step": train_stage, "global_step": global_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")

def main():
  global global_step
  global iteration

  crit = Critic(settings.IMG_CH, settings.TARGET_IMAGE_SIZE, settings.BASE_FEATURES, settings.FEATURES_MAX).to(settings.device)
  gen = Generator(settings.IMG_CH, settings.TARGET_IMAGE_SIZE, settings.BASE_FEATURES, settings.FEATURES_MAX, settings.Z_DIM).to(settings.device)

  opt_generator = optim.Adam(gen.parameters(), lr=settings.LR, betas=(0.0, 0.99))
  opt_critic = optim.Adam(crit.parameters(), lr=settings.LR, betas=(0.0, 0.99))

  summary_writer = SummaryWriter(f"logs/{settings.MODEL_NAME}", max_queue=50)
  batch_repeats = max(settings.BATCH_REPEATS if settings.BATCH_REPEATS is not None else 1, 1)

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

  train_stage = 0
  alpha = settings.START_ALPHA
  test_noise = torch.randn((settings.TESTING_SAMPLES, settings.Z_DIM, 1, 1), device=settings.device)
  if metadata is not None:
    if "iteration" in metadata.keys():
      iteration = metadata["iteration"]

    if "global_step" in metadata.keys():
      global_step = metadata["global_step"]

    if "train_step" in metadata.keys():
      train_stage = metadata["train_step"]

    if "alpha" in metadata.keys():
      alpha = metadata["alpha"]

    if "noise" in metadata.keys():
      tmp_noise = torch.Tensor(metadata["noise"])
      if tmp_noise.shape == (settings.TESTING_SAMPLES, settings.Z_DIM, 1, 1):
        test_noise = tmp_noise.to(settings.device)

  if settings.OVERRIDE_ITERATION is not None:
    print(f"[WARNING] Setting iteration to: {settings.OVERRIDE_ITERATION}")
    iteration = settings.OVERRIDE_ITERATION

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

  try:
    for iteraions_tuple in settings.PROGRESSIVE_ITERATIONS[train_stage:]:
      train_loop(iteraions_tuple[0], iteraions_tuple[1], train_stage, alpha, crit, gen, opt_critic, opt_generator, c_scaler, g_scaler, summary_writer, test_noise, batch_repeats)

      iteration = 0
      alpha = settings.START_ALPHA
      train_stage += 1

    print("Starting additional training")
    train_loop(0, settings.ADDITIONAL_TRAINING, len(settings.PROGRESSIVE_ITERATIONS) - 1, 1.0, crit, gen, opt_critic, opt_generator, c_scaler, g_scaler, summary_writer, test_noise, batch_repeats)
  except KeyboardInterrupt:
    print("Exiting")
  except Exception as e:
    print(e)
    try:
      save_metadata({"iteration": iteration, "train_step": train_stage, "global_step": global_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")
    except:
      pass
    exit(-1)

  save_model(gen, opt_generator, f"models/{settings.MODEL_NAME}/gen.mod")
  save_model(crit, opt_critic, f"models/{settings.MODEL_NAME}/crit.mod")
  save_metadata({"iteration": iteration, "train_step": train_stage, "global_step": global_step, "alpha": alpha, "noise": test_noise.tolist()}, f"models/{settings.MODEL_NAME}/metadata.pkl")

if __name__ == '__main__':
  main()
