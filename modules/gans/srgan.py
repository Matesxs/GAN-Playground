from colorama import Fore
import os
from typing import Union
import tensorflow as tf
import keras.backend as K
from keras import losses
from keras.optimizers import Optimizer, Adam
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.engine.network import Network
from keras.applications.vgg19 import VGG19, preprocess_input
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
from modules.helpers import time_to_format, get_paths_of_files_from_path
from settings.srgan_settings import RESTORE_BEST_PNSR_MODELS_EPISODES

# Calculate start image size based on final image size and number of upscales
def count_upscaling_start_size(target_image_shape: tuple, num_of_upscales: int):
  upsc = (target_image_shape[0] // (2 ** num_of_upscales), target_image_shape[1] // (2 ** num_of_upscales), target_image_shape[2])
  if upsc[0] < 1 or upsc[1] < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
  return upsc

def preprocess_vgg(x):
  """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
  if isinstance(x, np.ndarray):
    return preprocess_input((x + 1) * 127.5)
  else:
    return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)

def build_vgg(image_shape):
  # Get the vgg network. Extract features from last conv layer
  vgg = VGG19(include_top=False, weights="imagenet", input_shape=image_shape)
  vgg.trainable = False
  for l in vgg.layers:
    l.trainable = False

  # Create model and compile
  model = Model(inputs=vgg.input, outputs=vgg.layers[20].output, name="vgg_feature_extractor")
  model.trainable = False
  return model

def PSNR(y_true, y_pred):
  """
  PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
  The equation is:
  PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

  Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
  """
  return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

GAN_LOSS = "mse" # losses.Huber(delta=0.3)
DISC_LOSS = "binary_crossentropy"

# GAN, percept (vgg)
LOSS_WEIGHTS = [0.003, 0.006]

class SRGAN:
  AGREGATE_STAT_INTERVAL = 20_000  # Interval of saving data for pretrains
  AGREGATE_STAT_DIVIDER = 4 # Divider for agregate stat interval for gan training
  RESET_SEEDS_INTERVAL = 20_000  # Interval of checking norm gradient value of combined model
  CHECKPOINT_SAVE_INTERVAL = 5_000  # Interval of saving checkpoint

  def __init__(self, dataset_path:str, num_of_upscales:int,
               gen_mod_name:str, disc_mod_name:str,
               training_progress_save_path:str,
               testing_dataset_path:str=None,
               generator_optimizer:Optimizer=Adam(0.0001, 0.9), discriminator_optimizer:Optimizer=Adam(0.0001, 0.9),
               generator_lr_schedule:Union[dict, None]=None, discriminator_lr_schedule:Union[dict, None]=None,
               discriminator_label_noise:float=None, discriminator_label_noise_decay:float=None, discriminator_label_noise_min:float=0.001,
               batch_size:int=4, testing_batch_size:int=32, buffered_batches:int=20,
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
    self.testing_batch_size = testing_batch_size
    assert self.testing_batch_size > 0, Fore.RED + "Invalid testing batch size" + Fore.RESET

    if start_episode < 0: start_episode = 0
    self.episode_counter = start_episode

    # Create array of input image paths
    self.train_data = get_paths_of_files_from_path(dataset_path)
    assert self.train_data, Fore.RED + "Training dataset is not loaded" + Fore.RESET

    self.testing_data = None
    if testing_dataset_path:
      self.testing_data = get_paths_of_files_from_path(testing_dataset_path)
      assert self.testing_data, Fore.RED + "Testing dataset is not loaded" + Fore.RESET

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

    # Define static vars
    self.kernel_initializer = RandomNormal(stddev=0.02)
    self.custom_hr_test_images_paths = custom_hr_test_images_paths
    if custom_hr_test_images_paths and all([os.path.exists(p) for p in custom_hr_test_images_paths]):
      self.progress_test_images_paths = custom_hr_test_images_paths
    else:
      self.progress_test_images_paths = [random.choice(self.train_data)]

    # Create batchmaker and start it
    self.batch_maker = BatchMaker(self.train_data, self.batch_size, buffered_batches=buffered_batches, secondary_size=self.start_image_shape, num_of_loading_workers=num_of_loading_workers)
    self.batch_maker.start()

    self.testing_batchmaker = None
    if self.testing_data:
      self.testing_batchmaker = BatchMaker(self.testing_data, self.testing_batch_size, buffered_batches=buffered_batches, secondary_size=self.start_image_shape, num_of_loading_workers=num_of_loading_workers)
      self.testing_batchmaker.start()

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
    if self.generator.output_shape[1:] != self.target_image_shape: raise Exception("Invalid image input size for this generator model")
    self.generator.compile(loss=GAN_LOSS, optimizer=generator_optimizer, metrics=[PSNR])

    #################################
    ###    Create vgg network     ###
    #################################
    self.vgg = build_vgg(self.target_image_shape)
    self.vgg.compile(loss=GAN_LOSS, optimizer=generator_optimizer, metrics=['accuracy'])

    #################################
    ### Create combined generator ###
    #################################
    small_image_input = Input(shape=self.start_image_shape, name="small_image_input")
    gen_images = self.generator(small_image_input)

    # Extracts features from generated image
    generated_features = self.vgg(preprocess_vgg(gen_images))

    # Discriminator takes images and determinates validity
    frozen_discriminator = Network(self.discriminator.inputs, self.discriminator.outputs, name="frozen_discriminator")
    frozen_discriminator.trainable = False
    validity = frozen_discriminator(gen_images)

    # Combine models
    # Train generator to fool discriminator
    self.combined_generator_model = Model(small_image_input, outputs=[validity, generated_features], name="srgan_model")
    self.combined_generator_model.compile(loss=[DISC_LOSS, GAN_LOSS],
                                          loss_weights=LOSS_WEIGHTS,
                                          optimizer=generator_optimizer)

    # Print all summaries
    print("\nDiscriminator Summary:")
    self.discriminator.summary()
    print("\nGenerator Summary:")
    self.generator.summary()
    print("\nGAN Summary")
    self.combined_generator_model.summary()

    # Stats
    self.pnsr_record = None

    # Load checkpoint
    self.initiated = False
    if load_from_checkpoint: self.__load_checkpoint()

    # Load weights from param and override checkpoint weights
    if generator_weights: self.generator.load_weights(generator_weights)
    if discriminator_weights: self.discriminator.load_weights(discriminator_weights)

    # Set LR
    new_gen_lr = self.gen_lr_scheduler.set_lr(self.generator)
    self.gen_lr_scheduler.set_lr(self.combined_generator_model)
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

      if self.testing_data:
        res = p.map(check_image, self.testing_data)
        if not all(res): raise Exception("Inconsistent testing dataset")

    print(Fore.BLUE + "Dataset valid" + Fore.RESET)

  # Create generator based on template selected by name
  def __build_generator(self, model_name:str):
    small_image_input = Input(shape=self.start_image_shape)

    try:
      m = getattr(upscaling_generator_models_spreadsheet, model_name)(small_image_input, self.start_image_shape, self.num_of_upscales, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Generator model not found!\n{e}")

    return Model(small_image_input, m, name="generator_model")

  # Create discriminator based on teplate selected by name
  def __build_discriminator(self, model_name:str, classification:bool=True):
    img = Input(shape=self.target_image_shape)

    try:
      m = getattr(discriminator_models_spreadsheet, model_name)(img, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Discriminator model not found!\n{e}")

    if classification:
      m = Dense(1, activation="sigmoid")(m)

    return Model(img, m, name="discriminator_model")

  def __train_generator(self):
    large_images, small_images = self.batch_maker.get_batch()
    self.generator.train_on_batch(small_images, large_images)

  def __train_discriminator(self, discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False):
    # Function for adding random noise to labels (flipping them)
    def noising_labels(labels: np.ndarray, noise_ammount: float = 0.01):
      array = np.zeros(labels.shape)
      for idx in range(labels.shape[0]):
        if random.random() < noise_ammount:
          array[idx] = 1 - labels[idx]
          if array[idx] < 0: array[idx] = -array[idx]
        else:
          array[idx] = labels[idx]
      return labels

    if discriminator_smooth_real_labels:
      drl = np.random.uniform(0.7, 1.2, size=(self.batch_size, 1))
    else:
      drl = np.ones(shape=(self.batch_size, 1))

    if discriminator_smooth_fake_labels:
      dfl = np.random.uniform(0, 0.2, size=(self.batch_size, 1))
    else:
      dfl = np.zeros(shape=(self.batch_size, 1))

    # Adding random noise to discriminator labels
    if self.discriminator_label_noise and self.discriminator_label_noise > 0:
      disc_real_labels = noising_labels(drl, self.discriminator_label_noise / 2)
      disc_fake_labels = noising_labels(dfl, self.discriminator_label_noise / 2)
    else:
      disc_real_labels = drl.copy()
      disc_fake_labels = dfl.copy()

    large_images, small_images = self.batch_maker.get_batch()
    gen_imgs = self.generator.predict(small_images)

    real_loss = self.discriminator.train_on_batch(large_images, disc_real_labels)
    fake_loss = self.discriminator.train_on_batch(gen_imgs, disc_fake_labels)
    return real_loss, fake_loss

  def __train_gan(self, generator_smooth_labels:bool=False):
    large_images, small_images = self.batch_maker.get_batch()
    if generator_smooth_labels:
      gen_labels = np.random.uniform(0.8, 1.0, size=(self.batch_size, 1))
    else:
      gen_labels = np.ones(shape=(self.batch_size, 1))
    predicted_features = self.vgg.predict(preprocess_vgg(large_images))

    self.combined_generator_model.train_on_batch(small_images, [gen_labels, predicted_features])

  def train(self, target_episode:int, generator_train_episodes:int=None, discriminator_train_episodes:int=None, discriminator_training_multiplier:int=1,
            progress_images_save_interval:int=None, save_raw_progress_images:bool=True, weights_save_interval:int=None,
            discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False,
            generator_smooth_labels:bool=False,
            save_only_best_pnsr_weights:bool=False):

    # Check arguments and input data
    assert target_episode > 0, Fore.RED + "Invalid number of episodes" + Fore.RESET
    assert discriminator_training_multiplier > 0, Fore.RED + "Invalid discriminator training multiplier" + Fore.RESET
    assert generator_train_episodes > 0 or generator_train_episodes is None, Fore.RED + "Invalid generator train episodes" + Fore.RESET
    assert discriminator_train_episodes > 0 or discriminator_train_episodes is None, Fore.RED + "Invalid discriminator train episodes" + Fore.RESET
    if progress_images_save_interval:
      assert progress_images_save_interval <= target_episode, Fore.RED + "Invalid progress save interval" + Fore.RESET
    if weights_save_interval:
      assert weights_save_interval <= target_episode, Fore.RED + "Invalid weights save interval" + Fore.RESET

    if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)

    # Calculate epochs to go
    if generator_train_episodes:
      target_episode += generator_train_episodes
    if discriminator_train_episodes:
      target_episode += discriminator_train_episodes
    end_episode = target_episode
    target_episode = target_episode - self.episode_counter
    assert target_episode > 0, Fore.CYAN + "Training is already finished" + Fore.RESET

    epochs_time_history = deque(maxlen=self.AGREGATE_STAT_INTERVAL * 50)
    training_state = "Standby"

    # Save starting kernels and biases
    if not self.initiated:
      self.__save_img(save_raw_progress_images)
      self.tensorboard.log_kernels_and_biases(self.generator)
      self.save_checkpoint()

    print(Fore.GREEN + f"Starting training on episode {self.episode_counter} for {target_episode} episode" + Fore.RESET)
    for _ in range(target_episode):
      ep_start = time.time()
      if generator_train_episodes and (self.episode_counter < generator_train_episodes):
        if training_state != "Generator Training":
          print(Fore.BLUE + "Starting generator training" + Fore.RESET)
          training_state = "Generator Training"

        # Pretrain generator
        self.__train_generator()

      elif discriminator_train_episodes and (discriminator_train_episodes + (generator_train_episodes if generator_train_episodes else 0)) > self.episode_counter >= (generator_train_episodes if generator_train_episodes else 0):
        if training_state != "Discriminator Training":
          print(Fore.BLUE + "Starting discriminator training" + Fore.RESET)
          training_state = "Discriminator Training"

        # Pretrain discriminator
        self.__train_discriminator(discriminator_smooth_real_labels, discriminator_smooth_fake_labels)

      elif self.episode_counter >= ((generator_train_episodes if generator_train_episodes else 0) + (discriminator_train_episodes if discriminator_train_episodes else 0)):
        if training_state != "GAN Training":
          if (training_state != "Standby") and self.pnsr_record:
            print(Fore.MAGENTA + "Reseting PNSR record" + Fore.RESET)
            self.pnsr_record = None
            self.save_checkpoint()

          print(Fore.BLUE + "Starting GAN training" + Fore.RESET)
          training_state = "GAN Training"

        ### Train Discriminator ###
        # Train discriminator (real as ones and fake as zeros)
        for _ in range(discriminator_training_multiplier):
          self.__train_discriminator(discriminator_smooth_real_labels, discriminator_smooth_fake_labels)

        ### Train GAN ###
        # Train GAN (wants discriminator to recognize fake images as valid)
        self.__train_gan(generator_smooth_labels)

      self.episode_counter += 1
      self.tensorboard.step = self.episode_counter
      self.gen_lr_scheduler.episode = self.episode_counter
      self.disc_lr_scheduler.episode = self.episode_counter

      # Set LR based on episode count and schedule
      new_gen_lr = self.gen_lr_scheduler.set_lr(self.generator)
      self.gen_lr_scheduler.set_lr(self.combined_generator_model)
      new_disc_lr = self.disc_lr_scheduler.set_lr(self.discriminator)

      if new_gen_lr: print(Fore.MAGENTA + f"New LR for generator is {new_gen_lr}" + Fore.RESET)
      if new_disc_lr: print(Fore.MAGENTA + f"New LR for discriminator is {new_disc_lr}" + Fore.RESET)
      if new_gen_lr or new_disc_lr: self.save_checkpoint()

      # Save stats and print them to console
      if self.episode_counter % (self.AGREGATE_STAT_INTERVAL if (training_state in ["Generator Training", "Discriminator Training"]) else (self.AGREGATE_STAT_INTERVAL // self.AGREGATE_STAT_DIVIDER)) == 0:
        print(Fore.LIGHTBLUE_EX + "Stat evaluation started" + Fore.RESET)

        testing_batchmaker = self.testing_batchmaker if self.testing_batchmaker else self.batch_maker
        stats_raw = deque()

        try:
          for _ in range(testing_batchmaker.get_number_of_batches_in_dataset()):
            large_images, small_images = testing_batchmaker.get_batch()

            gen_imgs = self.generator.predict(small_images)

            # Evaluate models state
            disc_real_loss = float(self.discriminator.test_on_batch(large_images, np.ones(shape=(large_images.shape[0], 1))))
            disc_fake_loss = float(self.discriminator.test_on_batch(gen_imgs, np.zeros(shape=(gen_imgs.shape[0], 1))))
            disc_loss = (disc_real_loss + disc_fake_loss) * 0.5

            predicted_features = self.vgg.predict(preprocess_vgg(large_images))

            gan_losses = self.combined_generator_model.test_on_batch(small_images, [np.ones(shape=(large_images.shape[0], 1)), predicted_features])
            g_loss , pnsr = self.generator.test_on_batch(small_images, large_images)

            stats_raw.append([disc_loss, disc_real_loss, disc_fake_loss, float(g_loss), float(pnsr)] + [float(l) for l in gan_losses])
        except KeyboardInterrupt:
          # On keyboard interrupt return one step back, save a checkpoint and create new keybord interrupt exception
          # This is done to prevent skipping the evaluating step
          print(Fore.YELLOW + "Stat evaluation interrupted" + Fore.RESET)
          self.episode_counter -= 1
          self.save_checkpoint()
          raise KeyboardInterrupt("Stat evaluation interrupted")

        mean_stats = np.mean(stats_raw, 0)
        min_stats = np.min(stats_raw, 0)
        max_stats = np.max(stats_raw, 0)
        del stats_raw

        if training_state in ["GAN Training", "Generator Training"]:
          if not self.pnsr_record:
            self.pnsr_record = {"episode": self.episode_counter, "value": mean_stats[4]}
          elif self.pnsr_record["value"] < mean_stats[4]:
            print(Fore.MAGENTA + f"New PNSR record <{round(mean_stats[4], 5)}> on episode {self.episode_counter}!" + Fore.RESET)
            self.pnsr_record = {"episode": self.episode_counter, "value": mean_stats[4]}
            self.__save_weights()

        self.tensorboard.log_kernels_and_biases(self.generator)
        self.tensorboard.update_stats(self.episode_counter, gen_lr=self.gen_lr_scheduler.lr, disc_lr=self.disc_lr_scheduler.lr, disc_loss=mean_stats[0], disc_real_loss=mean_stats[1], disc_fake_loss=mean_stats[2], gan_loss=mean_stats[5], gen_loss=mean_stats[3], pnsr_mean=mean_stats[4], pnsr_min=min_stats[4], pnsr_max=max_stats[4], disc_label_noise=self.discriminator_label_noise if self.discriminator_label_noise else 0)

        print(Fore.GREEN + f"{self.episode_counter}/{end_episode}, Remaining: {(time_to_format(mean(epochs_time_history) * (end_episode - self.episode_counter))) if epochs_time_history else 'Unable to calculate'}, State: <{training_state}>\t\t[D Loss: {round(mean_stats[0], 5)}, D-R loss: {round(mean_stats[1], 5)}, D-F loss: {round(mean_stats[2], 5)}] [G loss: {round(mean_stats[3], 5)}, PNSR: [Min: {round(min_stats[4], 3)}db, Mean: {round(mean_stats[4], 3)}db, Max: {round(max_stats[4], 3)}db]] [GAN loss: {round(mean_stats[5], 5)}, Separated losses: {mean_stats[6:]}] - Epsilon: {round(self.discriminator_label_noise, 4) if self.discriminator_label_noise else 0}" + Fore.RESET)
        if self.pnsr_record:
          print(Fore.GREEN + f"Actual PNSR Record: {round(self.pnsr_record['value'], 5)} on episode {self.pnsr_record['episode']}" + Fore.RESET)

      # Decay label noise
      if self.discriminator_label_noise and self.discriminator_label_noise_decay:
        if training_state in ["GAN Training", "Discriminator Training"]:
          self.discriminator_label_noise = max([self.discriminator_label_noise_min, (self.discriminator_label_noise * self.discriminator_label_noise_decay)])

          if self.discriminator_label_noise != self.discriminator_label_noise_min and (self.discriminator_label_noise - self.discriminator_label_noise_min) < 0.001:
            self.discriminator_label_noise = self.discriminator_label_noise_min

      # Save progress
      if progress_images_save_interval is not None and self.episode_counter % progress_images_save_interval == 0 and training_state != "Discriminator Training":
        self.__save_img(save_raw_progress_images)

      # Save weights of models
      if weights_save_interval is not None and self.episode_counter % weights_save_interval == 0 and training_state in ["GAN Training", "Generator Training"] and not save_only_best_pnsr_weights:
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
      if (self.episode_counter in RESTORE_BEST_PNSR_MODELS_EPISODES) and self.pnsr_record:
        self.load_gen_weights_from_episode(self.pnsr_record["episode"])
        self.load_disc_weights_from_episode(self.pnsr_record["episode"])
        self.pnsr_record = None
        self.save_checkpoint()
        print(Fore.MAGENTA + "Best models weights restored and PNSR record restarted" + Fore.RESET)

      epochs_time_history.append(time.time() - ep_start)

    # Shutdown helper threads
    print(Fore.GREEN + "Training Complete - Waiting for other threads to finish" + Fore.RESET)
    if self.testing_batchmaker: self.testing_batchmaker.terminate = True
    self.batch_maker.terminate = True
    self.save_checkpoint()
    if not save_only_best_pnsr_weights:
      self.__save_weights()
    self.batch_maker.join()
    if self.testing_batchmaker: self.testing_batchmaker.join()
    print(Fore.GREEN + "All threads finished" + Fore.RESET)
    print(Fore.MAGENTA + f"PNSR Record: {round(self.pnsr_record['value'], 5)} on episode {self.pnsr_record['episode']}" + Fore.RESET)

  def __save_img(self, save_raw_progress_images:bool=True):
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
    self.tensorboard.write_image(np.reshape(cv.cvtColor(final_image, cv.COLOR_BGR2RGB) / 255, (-1, final_image.shape[0], final_image.shape[1], final_image.shape[2])).astype(np.float32))

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

        if "pnsr_record" in data.keys():
          if data["pnsr_record"]:
            self.pnsr_record = data["pnsr_record"]

        if "gen_lr" in data.keys():
          if data["gen_lr"]:
            self.gen_lr_scheduler.lr = data["gen_lr"]

        if "disc_lr" in data.keys():
          if data["disc_lr"]:
            self.disc_lr_scheduler.lr = data["disc_lr"]

        if not self.custom_hr_test_images_paths:
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
      "pnsr_record": self.pnsr_record,
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