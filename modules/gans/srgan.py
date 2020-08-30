from colorama import Fore
import os
from typing import Union
import keras.backend as K
from keras.optimizers import Optimizer, Adam
from keras.layers import Input, Dense
from keras.models import Model
from keras.engine.network import Network
from keras.initializers import RandomNormal
from keras.utils import plot_model
from statistics import mean
from PIL import Image
import numpy as np
import cv2 as cv
from collections import deque
import json
import random
import time
import imagesize
from multiprocessing.pool import ThreadPool

from modules.models import upscaling_generator_models_spreadsheet, discriminator_models_spreadsheet
from modules.keras_extensions.custom_tensorboard import TensorBoardCustom
from modules.keras_extensions.custom_lrscheduler import LearningRateScheduler
from modules.batch_maker import BatchMaker
from modules.stat_logger import StatLogger
from modules.helpers import time_to_format, get_paths_of_files_from_path, count_upscaling_start_size
from modules.keras_extensions.feature_extractor import create_feature_extractor, preprocess_vgg
from settings.srgan_settings import RESTORE_BEST_PNSR_MODELS_EPISODES

def PSNR(y_true, y_pred):
  """
  PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
  The equation is:
  PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

  Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
  """
  return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

def RGB_to_Y(image):
  R = image[:, :, :, 0]
  G = image[:, :, :, 1]
  B = image[:, :, :, 2]

  Y = 16 + (65.738 * R) + 129.057 * G + 25.064 * B
  return Y / 255.0

def PSNR_Y(y_true, y_pred):
  y_true = RGB_to_Y(y_true)
  y_pred = RGB_to_Y(y_pred)
  return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

GEN_LOSS = "mae"
DISC_LOSS = "binary_crossentropy"
FEATURE_LOSS = "mse"

FEATURE_EXTRACTOR_LAYERS = [5, 9]

GEN_LOSS_WEIGHT = 1.0 # 0.8
DISC_LOSS_WEIGHT = 0 # 0.01
FEATURE_LOSS_WEIGHTS = [0, 0] # [0.0415, 0.0415]

class SRGAN:
  SHOW_STATS_INTERVAL = 200  # Interval of saving data for pretrains
  RESET_SEEDS_INTERVAL = 5_000  # Interval of checking norm gradient value of combined model
  CHECKPOINT_SAVE_INTERVAL = 500  # Interval of saving checkpoint

  def __init__(self, dataset_path:str, num_of_upscales:int,
               gen_mod_name:str, disc_mod_name:str,
               training_progress_save_path:str,
               generator_optimizer:Optimizer=Adam(0.0001, 0.9), discriminator_optimizer:Optimizer=Adam(0.0001, 0.9),
               generator_lr_schedule:Union[dict, None]=None, discriminator_lr_schedule:Union[dict, None]=None,
               discriminator_label_noise:float=None, discriminator_label_noise_decay:float=None, discriminator_label_noise_min:float=0.001,
               batch_size:int=4, buffered_batches:int=20,
               generator_weights:Union[str, None]=None, discriminator_weights:Union[str, None]=None,
               start_episode:int=0, load_from_checkpoint:bool=False,
               custom_hr_test_images_paths:list=None, check_dataset:bool=True, num_of_loading_workers:int=8):

    self.disc_mod_name = disc_mod_name
    self.gen_mod_name = gen_mod_name
    self.num_of_upscales = num_of_upscales
    assert self.num_of_upscales >= 0, Fore.RED + "Invalid number of upscales" + Fore.RESET

    self.discriminator_label_noise = discriminator_label_noise
    self.discriminator_label_noise_decay = discriminator_label_noise_decay
    self.discriminator_label_noise_min = discriminator_label_noise_min

    self.batch_size = batch_size
    assert self.batch_size > 0, Fore.RED + "Invalid batch size" + Fore.RESET

    if start_episode < 0: start_episode = 0
    self.episode_counter = start_episode

    # Create array of input image paths
    self.train_data = get_paths_of_files_from_path(dataset_path)
    assert self.train_data, Fore.RED + "Training dataset is not loaded" + Fore.RESET

    # Load one image to get shape of it
    self.target_image_shape = cv.imread(self.train_data[0]).shape

    # Check image size validity
    if self.target_image_shape[0] < 4 or self.target_image_shape[1] < 4: raise Exception("Images too small, min size (4, 4)")

    # Starting image size calculate
    self.start_image_shape = count_upscaling_start_size(self.target_image_shape, self.num_of_upscales)

    # Check validity of whole datasets
    if check_dataset:
      self.__validate_dataset()

    # Initialize training data folder and logging
    self.training_progress_save_path = training_progress_save_path
    self.training_progress_save_path = os.path.join(self.training_progress_save_path, f"{self.gen_mod_name}__{self.disc_mod_name}__{self.start_image_shape}_to_{self.target_image_shape}")
    self.tensorboard = TensorBoardCustom(log_dir=os.path.join(self.training_progress_save_path, "logs"))
    self.stat_logger = StatLogger(self.tensorboard)

    # Define static vars
    self.kernel_initializer = RandomNormal(stddev=0.02)
    self.custom_hr_test_images_paths = custom_hr_test_images_paths
    self.custom_loading_failed = False
    if custom_hr_test_images_paths:
      self.progress_test_images_paths = custom_hr_test_images_paths
      for idx, image_path in enumerate(self.progress_test_images_paths):
        if not os.path.exists(image_path):
          self.custom_loading_failed = True
          self.progress_test_images_paths[idx] = random.choice(self.train_data)
    else:
      self.progress_test_images_paths = [random.choice(self.train_data)]

    # Create batchmaker and start it
    self.batch_maker = BatchMaker(self.train_data, self.batch_size, buffered_batches=buffered_batches, secondary_size=self.start_image_shape, num_of_loading_workers=num_of_loading_workers)

    # Create LR Schedulers for both "Optimizer"
    self.gen_lr_scheduler = LearningRateScheduler(lr_plan=generator_lr_schedule, start_lr=float(K.get_value(generator_optimizer.lr)))
    self.disc_lr_scheduler = LearningRateScheduler(lr_plan=discriminator_lr_schedule, start_lr=float(K.get_value(discriminator_optimizer.lr)))

    #################################
    ###   Create discriminator    ###
    #################################
    self.discriminator = self.__build_discriminator(disc_mod_name)
    self.discriminator.compile(loss=DISC_LOSS, optimizer=discriminator_optimizer)

    #################################
    ###     Create generator      ###
    #################################
    self.generator = self.__build_generator(gen_mod_name)
    if self.generator.output_shape[1:] != self.target_image_shape: raise Exception(f"Invalid image input size for this generator model\nGenerator shape: {self.generator.output_shape[1:]}, Target shape: {self.target_image_shape}")
    self.generator.compile(loss=GEN_LOSS, optimizer=generator_optimizer, metrics=[PSNR])

    #################################
    ###    Create vgg network     ###
    #################################
    self.vgg = create_feature_extractor(self.target_image_shape, FEATURE_EXTRACTOR_LAYERS)

    #################################
    ### Create combined generator ###
    #################################
    small_image_input = Input(shape=self.start_image_shape, name="small_image_input")

    # Images upscaled by generator
    gen_images = self.generator(small_image_input)

    # Extracts features from generated images
    generated_features = self.vgg(preprocess_vgg(gen_images))

    # Discriminator takes images and determinates validity
    frozen_discriminator = Network(self.discriminator.inputs, self.discriminator.outputs, name="frozen_discriminator")
    frozen_discriminator.trainable = False

    validity = frozen_discriminator(gen_images)

    # Combine models
    # Train generator to fool discriminator
    self.combined_generator_model = Model(small_image_input, outputs=[gen_images, validity] + [*generated_features], name="srgan")
    self.combined_generator_model.compile(loss=[GEN_LOSS, DISC_LOSS] + ([FEATURE_LOSS] * len(generated_features)),
                                          loss_weights=[GEN_LOSS_WEIGHT, DISC_LOSS_WEIGHT] + FEATURE_LOSS_WEIGHTS,
                                          optimizer=generator_optimizer, metrics={"generator": [PSNR_Y, PSNR]})

    # Print all summaries
    print("\nDiscriminator Summary:")
    self.discriminator.summary()
    print("\nGenerator Summary:")
    self.generator.summary()
    print("\nGAN Summary")
    self.combined_generator_model.summary()

    # Stats
    self.psnr_record = None

    # Load checkpoint
    self.initiated = False
    if load_from_checkpoint: self.__load_checkpoint()

    # Load weights from param and override checkpoint weights
    if generator_weights: self.generator.load_weights(generator_weights)
    if discriminator_weights: self.discriminator.load_weights(discriminator_weights)

    # Set LR
    # new_gen_lr = self.gen_lr_scheduler.set_lr(self.generator)
    new_gen_lr = self.gen_lr_scheduler.set_lr(self.combined_generator_model)
    new_disc_lr = self.disc_lr_scheduler.set_lr(self.discriminator)
    if new_gen_lr or new_disc_lr: self.save_checkpoint()
    if new_gen_lr: print(Fore.MAGENTA + f"New LR for generator is {new_gen_lr}" + Fore.RESET)
    if new_disc_lr: print(Fore.MAGENTA + f"New LR for discriminator is {new_disc_lr}" + Fore.RESET)

  # Check if datasets have consistent shapes
  def __validate_dataset(self):
    def check_image(image_path):
      im_shape = imagesize.get(image_path)
      if im_shape[0] != self.target_image_shape[0] or im_shape[1] != self.target_image_shape[1]:
        return False
      return True

    print(Fore.BLUE + "Checking dataset validity" + Fore.RESET)
    with ThreadPool(processes=8) as p:
      res = p.map(check_image, self.train_data)
      if not all(res): raise Exception("Inconsistent training dataset")

    print(Fore.BLUE + "Dataset valid" + Fore.RESET)

  # Create generator based on template selected by name
  def __build_generator(self, model_name:str):
    small_image_input = Input(shape=self.start_image_shape)

    try:
      m = getattr(upscaling_generator_models_spreadsheet, model_name)(small_image_input, self.start_image_shape, self.num_of_upscales, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Generator model not found!\n{e}")

    return Model(small_image_input, m, name="generator")

  # Create discriminator based on teplate selected by name
  def __build_discriminator(self, model_name:str, classification:bool=True):
    img = Input(shape=self.target_image_shape)

    try:
      m = getattr(discriminator_models_spreadsheet, model_name)(img, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Discriminator model not found!\n{e}")

    if classification:
      m = Dense(1, activation="sigmoid")(m)

    return Model(img, m, name="discriminator")

  def __train_discriminator(self, discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False):
    if discriminator_smooth_real_labels:
      disc_real_labels = np.random.uniform(0.7, 1.2, size=(self.batch_size, 1))
    else:
      disc_real_labels = np.ones(shape=(self.batch_size, 1))

    if discriminator_smooth_fake_labels:
      disc_fake_labels = np.random.uniform(0, 0.2, size=(self.batch_size, 1))
    else:
      disc_fake_labels = np.zeros(shape=(self.batch_size, 1))

    # Adding random noise to discriminator labels
    if self.discriminator_label_noise and self.discriminator_label_noise > 0:
      disc_real_labels += (np.random.uniform(size=(self.batch_size, 1)) * (self.discriminator_label_noise / 2))
      disc_fake_labels += (np.random.uniform(size=(self.batch_size, 1)) * (self.discriminator_label_noise / 2))

    large_images, small_images = self.batch_maker.get_batch()
    gen_imgs = self.generator.predict(small_images)

    real_loss = self.discriminator.train_on_batch(large_images, disc_real_labels)
    fake_loss = self.discriminator.train_on_batch(gen_imgs, disc_fake_labels)
    return real_loss, fake_loss

  def __train_gan(self, generator_smooth_labels:bool=False):
    large_images, small_images = self.batch_maker.get_batch()
    if generator_smooth_labels:
      valid_labels = np.random.uniform(0.8, 1.0, size=(self.batch_size, 1))
    else:
      valid_labels = np.ones(shape=(self.batch_size, 1))
    predicted_features = self.vgg.predict(preprocess_vgg(large_images))

    gan_metrics = self.combined_generator_model.train_on_batch(small_images, [large_images, valid_labels] + predicted_features)

    return float(gan_metrics[0]), [float(x) for x in gan_metrics[1:-2]], float(gan_metrics[-1]), float(gan_metrics[-2])

  def train(self, target_episode:int, discriminator_training_multiplier:int=1,
            progress_images_save_interval:int=None, save_raw_progress_images:bool=True, weights_save_interval:int=None,
            discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False,
            generator_smooth_labels:bool=False,
            save_only_best_pnsr_weights:bool=False):

    # Check arguments and input data
    assert target_episode > 0, Fore.RED + "Invalid number of episodes" + Fore.RESET
    assert discriminator_training_multiplier > 0, Fore.RED + "Invalid discriminator training multiplier" + Fore.RESET
    if progress_images_save_interval:
      assert progress_images_save_interval <= target_episode, Fore.RED + "Invalid progress save interval" + Fore.RESET
    if weights_save_interval:
      assert weights_save_interval <= target_episode, Fore.RED + "Invalid weights save interval" + Fore.RESET

    if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)

    # Calculate epochs to go
    episodes_to_go = target_episode - self.episode_counter
    assert episodes_to_go > 0, Fore.CYAN + "Training is already finished" + Fore.RESET

    epochs_time_history = deque(maxlen=self.SHOW_STATS_INTERVAL * 50)

    # Save starting kernels and biases
    if not self.initiated:
      self.__save_img(save_raw_progress_images)
      self.save_checkpoint()

    print(Fore.GREEN + f"Starting training on episode {self.episode_counter} for {target_episode} episode" + Fore.RESET)
    print(Fore.MAGENTA + "Preview training stats in tensorboard: http://localhost:6006" + Fore.RESET)
    for _ in range(episodes_to_go):
      ep_start = time.time()

      ### Train Discriminator ###
      # Train discriminator (real as ones and fake as zeros)
      disc_stats = deque(maxlen=discriminator_training_multiplier)

      for _ in range(discriminator_training_multiplier):
        real_loss, fake_loss = self.__train_discriminator(discriminator_smooth_real_labels, discriminator_smooth_fake_labels)
        disc_stats.append([real_loss, fake_loss])

      disc_stats = np.mean(disc_stats, 0)
      disc_loss = sum(disc_stats) * 0.5

      ### Train GAN ###
      # Train GAN (wants discriminator to recognize fake images as valid)
      gen_loss, partial_gan_losses, psnr, psnr_y = self.__train_gan(generator_smooth_labels)

      self.stat_logger.append_stats(self.episode_counter, disc_loss=disc_loss, disc_real_loss=disc_stats[0], disc_fake_loss=disc_stats[1], gen_loss=gen_loss, psnr=psnr, psnr_y=psnr_y, disc_label_noise=self.discriminator_label_noise if self.discriminator_label_noise else 0)

      if psnr_y > 15:
        if not self.psnr_record:
          self.psnr_record = {"episode": self.episode_counter, "value": psnr_y}
        elif self.psnr_record["value"] < psnr_y:
          print(Fore.MAGENTA + f"New PSNR (PSNR_Y) record <{round(psnr_y, 5)}> on episode {self.episode_counter}!" + Fore.RESET)
          self.psnr_record = {"episode": self.episode_counter, "value": psnr_y}
          self.__save_weights()
          self.__save_img(False, tensorflow_description="best_psnr_images")

      self.episode_counter += 1
      self.tensorboard.step = self.episode_counter
      self.gen_lr_scheduler.episode = self.episode_counter
      self.disc_lr_scheduler.episode = self.episode_counter

      # Set LR based on episode count and schedule
      # new_gen_lr = self.gen_lr_scheduler.set_lr(self.generator)
      new_gen_lr = self.gen_lr_scheduler.set_lr(self.combined_generator_model)
      new_disc_lr = self.disc_lr_scheduler.set_lr(self.discriminator)

      if new_gen_lr: print(Fore.MAGENTA + f"New LR for generator is {new_gen_lr}" + Fore.RESET)
      if new_disc_lr: print(Fore.MAGENTA + f"New LR for discriminator is {new_disc_lr}" + Fore.RESET)
      if new_gen_lr or new_disc_lr: self.save_checkpoint()

      # Save stats and print them to console
      if self.episode_counter % self.SHOW_STATS_INTERVAL == 0:
        print(Fore.GREEN + f"{self.episode_counter}/{target_episode}, Remaining: {(time_to_format(mean(epochs_time_history) * (target_episode - self.episode_counter))) if epochs_time_history else 'Unable to calculate'}\t\tDiscriminator: [loss: {round(disc_loss, 5)}, real_loss: {round(float(disc_stats[0]), 5)}, fake_loss: {round(float(disc_stats[1]), 5)}, label_noise: {round(self.discriminator_label_noise * 100, 2) if self.discriminator_label_noise else 0}%] Generator: [loss: {round(gen_loss, 5)}, partial_losses: {partial_gan_losses}, psnr: {round(psnr, 3)}dB, psnr_y: {round(psnr_y, 3)}dB]" + Fore.RESET)
        if self.psnr_record:
          print(Fore.GREEN + f"Actual PSNR (PSNR_Y) Record: {round(self.psnr_record['value'], 5)} on episode {self.psnr_record['episode']}" + Fore.RESET)

      # Decay label noise
      if self.discriminator_label_noise and self.discriminator_label_noise_decay:
        self.discriminator_label_noise = max([self.discriminator_label_noise_min, (self.discriminator_label_noise * self.discriminator_label_noise_decay)])

        if self.discriminator_label_noise != self.discriminator_label_noise_min and (self.discriminator_label_noise - self.discriminator_label_noise_min) < 0.001:
          self.discriminator_label_noise = self.discriminator_label_noise_min

      # Save progress
      if progress_images_save_interval is not None and self.episode_counter % progress_images_save_interval == 0:
        self.__save_img(save_raw_progress_images)

      # Save weights of models
      if weights_save_interval is not None and self.episode_counter % weights_save_interval == 0 and not save_only_best_pnsr_weights:
        self.__save_weights()

      # Save checkpoint
      if self.episode_counter % self.CHECKPOINT_SAVE_INTERVAL == 0:
        self.save_checkpoint()
        print(Fore.BLUE + "Checkpoint created" + Fore.RESET)

      # Reset seeds
      if self.episode_counter % self.RESET_SEEDS_INTERVAL == 0:
        np.random.seed(None)
        random.seed()

      # Restore models with best pnsr and restart record
      if (self.episode_counter in RESTORE_BEST_PNSR_MODELS_EPISODES) and self.psnr_record:
        self.load_gen_weights_from_episode(self.psnr_record["episode"])
        self.load_disc_weights_from_episode(self.psnr_record["episode"])
        self.psnr_record = None
        self.save_checkpoint()
        print(Fore.MAGENTA + "Best models weights restored and PNSR record restarted" + Fore.RESET)

      epochs_time_history.append(time.time() - ep_start)

    # Shutdown helper threads
    print(Fore.GREEN + "Training Complete - Waiting for other threads to finish" + Fore.RESET)
    self.stat_logger.terminate()
    self.batch_maker.terminate()
    self.save_checkpoint()
    if not save_only_best_pnsr_weights:
      self.__save_weights()
    self.batch_maker.join()
    self.stat_logger.join()
    print(Fore.GREEN + "All threads finished" + Fore.RESET)
    print(Fore.MAGENTA + f"PSNR (PSNR_Y) Record: {round(self.psnr_record['value'], 5)} on episode {self.psnr_record['episode']}" + Fore.RESET)

  def __save_img(self, save_raw_progress_images:bool=True, tensorflow_description:str="progress"):
    if not os.path.exists(self.training_progress_save_path + "/progress_images"): os.makedirs(self.training_progress_save_path + "/progress_images")

    final_image = np.zeros(shape=(self.target_image_shape[0] * len(self.progress_test_images_paths), self.target_image_shape[1] * 3, self.target_image_shape[2])).astype(np.float32)

    for idx, test_image_path in enumerate(self.progress_test_images_paths):
      if not os.path.exists(test_image_path):
        print(Fore.YELLOW + f"Failed to locate test image: {test_image_path}, replacing it with new one!" + Fore.RESET)
        self.progress_test_images_paths[idx] = random.choice(self.train_data)
        self.save_checkpoint()

      # Load image for upscale and resize it to starting (small) image size
      original_unscaled_image = cv.imread(test_image_path)
      # print(f"[DEBUG] {original_unscaled_image.shape}, {self.target_image_shape}")
      if original_unscaled_image.shape != self.target_image_shape:
        original_image = cv.resize(original_unscaled_image, dsize=(self.start_image_shape[0], self.start_image_shape[1]), interpolation=(cv.INTER_AREA if (original_unscaled_image.shape[0] > self.start_image_shape[0] and original_unscaled_image.shape[1] > self.start_image_shape[1]) else cv.INTER_CUBIC))
      else:
        original_image = original_unscaled_image
      small_image = cv.resize(original_image, dsize=(self.start_image_shape[0], self.start_image_shape[1]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > self.start_image_shape[0] and original_image.shape[1] > self.start_image_shape[1]) else cv.INTER_CUBIC))

      # Conver image to RGB colors and upscale it
      gen_img = self.generator.predict(np.array([cv.cvtColor(small_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0]))[0]

      # Rescale images 0 to 255
      gen_img = (0.5 * gen_img + 0.5) * 255
      gen_img = cv.cvtColor(gen_img, cv.COLOR_RGB2BGR)

      # Place side by side image resized by opencv, original (large) image and upscaled by gan
      final_image[idx * gen_img.shape[1]:(idx + 1) * gen_img.shape[1], 0:gen_img.shape[0], :] = cv.resize(small_image, dsize=(self.target_image_shape[0], self.target_image_shape[1]), interpolation=(cv.INTER_AREA if (small_image.shape[0] > self.target_image_shape[0] and small_image.shape[1] > self.target_image_shape[1]) else cv.INTER_CUBIC))
      final_image[idx * gen_img.shape[1]:(idx + 1) * gen_img.shape[1], gen_img.shape[0]:gen_img.shape[0] * 2, :] = original_image
      final_image[idx * gen_img.shape[1]:(idx + 1) * gen_img.shape[1], gen_img.shape[0] * 2:gen_img.shape[0] * 3, :] = gen_img

    # Save image to folder and to tensorboard
    if save_raw_progress_images:
      cv.imwrite(f"{self.training_progress_save_path}/progress_images/{self.episode_counter}.png", final_image)
    self.tensorboard.write_image(np.reshape(cv.cvtColor(final_image, cv.COLOR_BGR2RGB) / 255, (-1, final_image.shape[0], final_image.shape[1], final_image.shape[2])).astype(np.float32), description=tensorflow_description)

  def __save_weights(self):
    save_dir = self.training_progress_save_path + "/weights/" + str(self.episode_counter)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")
    self.discriminator.save_weights(f"{save_dir}/discriminator_{self.disc_mod_name}.h5")

  def load_gen_weights_from_episode(self, episode:int):
    weights_dir = self.training_progress_save_path + "/weights/" + str(episode)
    if not os.path.exists(weights_dir): return

    gen_weights_path = weights_dir + f"/generator_{self.gen_mod_name}.h5"
    if os.path.exists(gen_weights_path):
      self.generator.load_weights(gen_weights_path)

  def load_disc_weights_from_episode(self, episode:int):
    weights_dir = self.training_progress_save_path + "/weights/" + str(episode)
    if not os.path.exists(weights_dir): return

    disc_weights_path = weights_dir + f"/discriminator_{self.disc_mod_name}.h5"
    if os.path.exists(disc_weights_path):
      self.discriminator.load_weights(disc_weights_path)

  def save_models_structure_images(self):
    save_path = self.training_progress_save_path + "/model_structures"
    if not os.path.exists(save_path): os.makedirs(save_path)
    plot_model(self.combined_generator_model, os.path.join(save_path, "combined.png"), expand_nested=True, show_shapes=True)
    plot_model(self.generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)
    plot_model(self.discriminator, os.path.join(save_path, "discriminator.png"), expand_nested=True, show_shapes=True)

  def __load_checkpoint(self):
    checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
    if not os.path.exists(os.path.join(checkpoint_base_path, "checkpoint_data.json")): return

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "rb") as f:
      data = json.load(f)

      if data:
        self.episode_counter = int(data["episode"])

        try:
          self.generator.load_weights(data["gen_path"])
        except:
          print(Fore.YELLOW + "Failed to load generator weights from checkpoint" + Fore.RESET)

        try:
          self.discriminator.load_weights(data["disc_path"])
        except:
          print(Fore.YELLOW + "Failed to load discriminator weights from checkpoint" + Fore.RESET)

        if "disc_label_noise" in data.keys():
          self.discriminator_label_noise = float(data["disc_label_noise"])

        if "psnr_record" in data.keys():
          if data["psnr_record"]:
            self.psnr_record = data["psnr_record"]

        if "gen_lr" in data.keys():
          if data["gen_lr"]:
            self.gen_lr_scheduler.lr = data["gen_lr"]

        if "disc_lr" in data.keys():
          if data["disc_lr"]:
            self.disc_lr_scheduler.lr = data["disc_lr"]

        if not self.custom_hr_test_images_paths or self.custom_loading_failed:
          self.progress_test_images_paths = data["test_image"]
        self.initiated = True

  def save_checkpoint(self):
    checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
    if not os.path.exists(checkpoint_base_path): os.makedirs(checkpoint_base_path)

    gen_path = f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5"
    disc_path = f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5"

    if os.path.exists(gen_path): os.rename(gen_path, f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5.lock")
    if os.path.exists(disc_path): os.rename(disc_path, f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5.lock")

    self.generator.save_weights(gen_path)
    self.discriminator.save_weights(disc_path)

    if os.path.exists(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5.lock")
    if os.path.exists(f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5.lock")

    data = {
      "episode": self.episode_counter,
      "gen_path": gen_path,
      "disc_path": disc_path,
      "disc_label_noise": self.discriminator_label_noise,
      "test_image": self.progress_test_images_paths,
      "psnr_record": self.psnr_record,
      "gen_lr": float(self.gen_lr_scheduler.lr),
      "disc_lr": float(self.disc_lr_scheduler.lr)
    }

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "w", encoding='utf-8') as f:
      json.dump(data, f)

  def make_progress_gif(self, frame_duration:int=16):
    if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)
    if not os.path.exists(self.training_progress_save_path + "/progress_images"): return

    frames = []
    img_file_names = os.listdir(self.training_progress_save_path + "/progress_images")

    for im_file in img_file_names:
      if os.path.isfile(self.training_progress_save_path + "/progress_images/" + im_file):
        frames.append(Image.open(self.training_progress_save_path + "/progress_images/" + im_file))

    if len(frames) > 2:
      frames[0].save(f"{self.training_progress_save_path}/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, optimize=False, duration=frame_duration, loop=0)