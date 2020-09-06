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
from cv2 import cv2 as cv
from collections import deque
import json
import random
import time
import imagesize
from multiprocessing.pool import ThreadPool

from ..models import upscaling_generator_models_spreadsheet, discriminator_models_spreadsheet
from ..keras_extensions.custom_tensorboard import TensorBoardCustom
from ..keras_extensions.custom_lrscheduler import LearningRateScheduler
from ..utils.batch_maker import BatchMaker, AugmentationSettings
from ..utils.stat_logger import StatLogger
from ..utils.helpers import time_to_format, get_paths_of_files_from_path, count_upscaling_start_size
from ..keras_extensions.feature_extractor import create_feature_extractor, preprocess_vgg
from ..utils.metrics import PSNR, PSNR_Y, SSIM

class SRGAN:
  SHOW_STATS_INTERVAL = 200  # Interval of saving data for pretrains
  RESET_SEEDS_INTERVAL = 5_000  # Interval of checking norm gradient value of combined model
  CHECKPOINT_SAVE_INTERVAL = 500  # Interval of saving checkpoint

  def __init__(self, dataset_path:str, num_of_upscales:int,
               gen_mod_name:str, disc_mod_name:str,
               training_progress_save_path:str,
               dataset_augmentation_settings:Union[AugmentationSettings, None]=None,
               generator_optimizer:Optimizer=Adam(0.0001, 0.9), discriminator_optimizer:Optimizer=Adam(0.0001, 0.9),
               gen_loss="mae", disc_loss="binary_crossentropy", feature_loss="mae",
               gen_loss_weight:float=1.0, disc_loss_weight:float=0.003, feature_loss_weights:Union[list, float, None]=None,
               feature_extractor_layers: Union[list, None]=None,
               generator_lr_decay_interval:Union[int, None]=None, discriminator_lr_decay_interval:Union[int, None]=None,
               generator_lr_decay_factor:Union[float, None]=None, discriminator_lr_decay_factor:Union[float, None]=None,
               generator_min_lr:Union[float, None]=None, discriminator_min_lr:Union[float, None]=None,
               discriminator_label_noise:Union[float, None]=None, discriminator_label_noise_decay:Union[float, None]=None, discriminator_label_noise_min:Union[float, None]=0.001,
               batch_size:int=4, buffered_batches:int=20,
               generator_weights:Union[str, None]=None, discriminator_weights:Union[str, None]=None,
               load_from_checkpoint:bool=False,
               custom_hr_test_images_paths:Union[list, None]=None, check_dataset:bool=True, num_of_loading_workers:int=8):

    # Save params to inner variables
    self.__disc_mod_name = disc_mod_name
    self.__gen_mod_name = gen_mod_name
    self.__num_of_upscales = num_of_upscales
    assert self.__num_of_upscales >= 0, Fore.RED + "Invalid number of upscales" + Fore.RESET

    self.__discriminator_label_noise = discriminator_label_noise
    self.__discriminator_label_noise_decay = discriminator_label_noise_decay
    self.__discriminator_label_noise_min = discriminator_label_noise_min
    if self.__discriminator_label_noise_min is None: self.__discriminator_label_noise_min = 0

    self.__batch_size = batch_size
    assert self.__batch_size > 0, Fore.RED + "Invalid batch size" + Fore.RESET

    self.__episode_counter = 0

    # Insert empty lists if feature extractor settings are empty
    if feature_extractor_layers is None:
      feature_extractor_layers = []

    if feature_loss_weights is None:
      feature_loss_weights = []

    # If feature_loss_weights is float then create list of the weights from it
    if isinstance(feature_loss_weights, float) and len(feature_extractor_layers) > 0:
      feature_loss_weights = [feature_loss_weights / len(feature_extractor_layers)] * len(feature_extractor_layers)
    else:
      assert len(feature_extractor_layers) == len(feature_loss_weights), Fore.RED + "Number of extractor layers and feature loss weights must match!" + Fore.RESET

    # Create array of input image paths
    self.__train_data = get_paths_of_files_from_path(dataset_path, only_files=True)
    assert self.__train_data, Fore.RED + "Training dataset is not loaded" + Fore.RESET

    # Load one image to get shape of it
    self.__target_image_shape = cv.imread(self.__train_data[0]).shape

    # Check image size validity
    if self.__target_image_shape[0] < 4 or self.__target_image_shape[1] < 4: raise Exception("Images too small, min size (4, 4)")

    # Starting image size calculate
    self.__start_image_shape = count_upscaling_start_size(self.__target_image_shape, self.__num_of_upscales)

    # Check validity of whole datasets
    if check_dataset:
      self.__validate_dataset()

    # Initialize training data folder and logging
    self.__training_progress_save_path = training_progress_save_path
    self.__training_progress_save_path = os.path.join(self.__training_progress_save_path, f"{self.__gen_mod_name}__{self.__disc_mod_name}__{self.__start_image_shape}_to_{self.__target_image_shape}")
    self.__tensorboard = TensorBoardCustom(log_dir=os.path.join(self.__training_progress_save_path, "logs"))
    self.__stat_logger = StatLogger(self.__tensorboard)

    # Define static vars
    self.kernel_initializer = RandomNormal(stddev=0.02)
    self.__custom_loading_failed = False
    self.__custom_test_images = True if custom_hr_test_images_paths else False
    if custom_hr_test_images_paths:
      self.__progress_test_images_paths = custom_hr_test_images_paths
      for idx, image_path in enumerate(self.__progress_test_images_paths):
        if not os.path.exists(image_path):
          self.__custom_loading_failed = True
          self.__progress_test_images_paths[idx] = random.choice(self.__train_data)
    else:
      self.__progress_test_images_paths = [random.choice(self.__train_data)]

    # Create batchmaker and start it
    self.batch_maker = BatchMaker(self.__train_data, self.__batch_size, buffered_batches=buffered_batches, secondary_size=self.__start_image_shape, num_of_loading_workers=num_of_loading_workers, augmentation_settings=dataset_augmentation_settings)

    # Create LR Schedulers for both "Optimizer"
    self.__gen_lr_scheduler = LearningRateScheduler(start_lr=float(K.get_value(generator_optimizer.lr)), lr_decay_factor=generator_lr_decay_factor, lr_decay_interval=generator_lr_decay_interval, min_lr=generator_min_lr)
    self.__disc_lr_scheduler = LearningRateScheduler(start_lr=float(K.get_value(discriminator_optimizer.lr)), lr_decay_factor=discriminator_lr_decay_factor, lr_decay_interval=discriminator_lr_decay_interval, min_lr=discriminator_min_lr)

    #####################################
    ###      Create discriminator     ###
    #####################################
    self.__discriminator = self.__build_discriminator(disc_mod_name)
    self.__discriminator.compile(loss=disc_loss, optimizer=discriminator_optimizer)

    #####################################
    ###       Create generator        ###
    #####################################
    self.__generator = self.__build_generator(gen_mod_name)
    if self.__generator.output_shape[1:] != self.__target_image_shape: raise Exception(f"Invalid image input size for this generator model\nGenerator shape: {self.__generator.output_shape[1:]}, Target shape: {self.__target_image_shape}")
    self.__generator.compile(loss=gen_loss, optimizer=generator_optimizer, metrics=[PSNR_Y, PSNR, SSIM])

    #####################################
    ###      Create vgg network       ###
    #####################################
    self.__vgg = create_feature_extractor(self.__target_image_shape, feature_extractor_layers)

    #####################################
    ### Create combined discriminator ###
    #####################################
    small_image_input_discriminator = Input(shape=self.__start_image_shape, name="small_image_input")
    large_image_input_discriminator = Input(shape=self.__target_image_shape, name="large_image_input")

    frozen_generator = Network(self.__generator.inputs, self.__generator.outputs, name="frozen_generator")
    frozen_generator.trainable = False

    upscaled_images = frozen_generator(small_image_input_discriminator)

    fake_validity = self.__discriminator(upscaled_images)
    real_validity = self.__discriminator(large_image_input_discriminator)

    self.__combined_discriminator_model = Model(inputs=[small_image_input_discriminator, large_image_input_discriminator], outputs=[fake_validity, real_validity], name="combined_discriminator")
    self.__combined_discriminator_model.compile(loss=disc_loss, optimizer=discriminator_optimizer, loss_weights=[1., 1.])

    #####################################
    ###   Create combined generator   ###
    #####################################
    small_image_input_generator = Input(shape=self.__start_image_shape, name="small_image_input")

    # Images upscaled by generator
    gen_images = self.__generator(small_image_input_generator)

    # Discriminator takes images and determinates validity
    frozen_discriminator = Network(self.__discriminator.inputs, self.__discriminator.outputs, name="frozen_discriminator")
    frozen_discriminator.trainable = False

    validity = frozen_discriminator(gen_images)

    # Extracts features from generated images
    generated_features = self.__vgg(preprocess_vgg(gen_images))

    # Combine models
    # Train generator to fool discriminator
    self.__combined_generator_model = Model(inputs=small_image_input_generator, outputs=[gen_images, validity] + [*generated_features], name="srgan")
    self.__combined_generator_model.compile(loss=[gen_loss, disc_loss] + ([feature_loss] * len(generated_features)),
                                            loss_weights=[gen_loss_weight, disc_loss_weight] + feature_loss_weights,
                                            optimizer=generator_optimizer, metrics={"generator": [PSNR_Y, PSNR, SSIM]})

    # Print all summaries
    print("\nDiscriminator Summary:")
    self.__discriminator.summary()
    print("\nGenerator Summary:")
    self.__generator.summary()

    # Load checkpoint
    self.__initiated = False
    if load_from_checkpoint: self.__load_checkpoint()

    # Load weights from param and override checkpoint weights
    if generator_weights: self.__generator.load_weights(generator_weights)
    if discriminator_weights: self.__discriminator.load_weights(discriminator_weights)

    # Set LR
    self.__gen_lr_scheduler.set_lr(self.__combined_generator_model, self.__episode_counter)
    self.__disc_lr_scheduler.set_lr(self.__discriminator, self.__episode_counter)

  @property
  def episode_counter(self):
    return self.__episode_counter

  # Check if datasets have consistent shapes
  def __validate_dataset(self):
    def check_image(image_path):
      im_shape = imagesize.get(image_path)
      if im_shape[0] != self.__target_image_shape[0] or im_shape[1] != self.__target_image_shape[1]:
        return False
      return True

    print(Fore.BLUE + "Checking dataset validity" + Fore.RESET)
    with ThreadPool(processes=8) as p:
      res = p.map(check_image, self.__train_data)
      if not all(res): raise Exception("Inconsistent training dataset")

    print(Fore.BLUE + "Dataset valid" + Fore.RESET)

  # Create generator based on template selected by name
  def __build_generator(self, model_name:str):
    small_image_input = Input(shape=self.__start_image_shape)

    try:
      m = getattr(upscaling_generator_models_spreadsheet, model_name)(small_image_input, self.__start_image_shape, self.__num_of_upscales, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Generator model not found!\n{e}")

    return Model(small_image_input, m, name="generator")

  # Create discriminator based on teplate selected by name
  def __build_discriminator(self, model_name:str, classification:bool=True):
    img = Input(shape=self.__target_image_shape)

    try:
      m = getattr(discriminator_models_spreadsheet, model_name)(img, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Discriminator model not found!\n{e}")

    if classification:
      m = Dense(1, activation="sigmoid")(m)

    return Model(img, m, name="discriminator")

  def __train_generator(self):
    large_images, small_images = self.batch_maker.get_batch()
    gen_loss, psnr_y, psnr, ssim = self.__generator.train_on_batch(small_images, large_images)
    return float(gen_loss), float(psnr), float(psnr_y), float(ssim)

  def __train_discriminator(self, discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False):
    if discriminator_smooth_real_labels:
      disc_real_labels = np.random.uniform(0.7, 1.2, size=(self.__batch_size, 1))
    else:
      disc_real_labels = np.ones(shape=(self.__batch_size, 1))

    if discriminator_smooth_fake_labels:
      disc_fake_labels = np.random.uniform(0, 0.2, size=(self.__batch_size, 1))
    else:
      disc_fake_labels = np.zeros(shape=(self.__batch_size, 1))

    # Adding random noise to discriminator labels
    if self.__discriminator_label_noise and self.__discriminator_label_noise > 0:
      disc_real_labels += (np.random.uniform(size=(self.__batch_size, 1)) * (self.__discriminator_label_noise / 2))
      disc_fake_labels += (np.random.uniform(size=(self.__batch_size, 1)) * (self.__discriminator_label_noise / 2))

    large_images, small_images = self.batch_maker.get_batch()

    disc_loss, disc_fake_loss, disc_real_loss = self.__combined_discriminator_model.train_on_batch([small_images, large_images], [disc_fake_labels, disc_real_labels])

    return float(disc_loss), float(disc_fake_loss), float(disc_real_loss)

  def __train_gan(self, generator_smooth_labels:bool=False):
    large_images, small_images = self.batch_maker.get_batch()
    if generator_smooth_labels:
      valid_labels = np.random.uniform(0.8, 1.0, size=(self.__batch_size, 1))
    else:
      valid_labels = np.ones(shape=(self.__batch_size, 1))
    predicted_features = self.__vgg.predict(preprocess_vgg(large_images))

    gan_metrics = self.__combined_generator_model.train_on_batch(small_images, [large_images, valid_labels] + predicted_features)

    return float(gan_metrics[0]), [round(float(x), 5) for x in gan_metrics[1:-3]], float(gan_metrics[-2]), float(gan_metrics[-3]), float(gan_metrics[-1])

  def train(self, target_episode:int, pretrain_episodes:Union[int, None]=None, discriminator_training_multiplier:int=1,
            progress_images_save_interval:Union[int, None]=None, save_raw_progress_images:bool=True, weights_save_interval:Union[int, None]=None,
            discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False,
            generator_smooth_labels:bool=False):

    # Check arguments and input data
    assert target_episode > 0, Fore.RED + "Invalid number of episodes" + Fore.RESET
    assert discriminator_training_multiplier > 0, Fore.RED + "Invalid discriminator training multiplier" + Fore.RESET
    if pretrain_episodes:
      assert pretrain_episodes <= target_episode, Fore.RED + "Pretrain episodes must be <= target episode" + Fore.RESET
    if progress_images_save_interval:
      assert progress_images_save_interval <= target_episode, Fore.RED + "Invalid progress save interval" + Fore.RESET
    if weights_save_interval:
      assert weights_save_interval <= target_episode, Fore.RED + "Invalid weights save interval" + Fore.RESET

    if not os.path.exists(self.__training_progress_save_path): os.makedirs(self.__training_progress_save_path)

    # Calculate epochs to go
    episodes_to_go = target_episode - self.__episode_counter
    assert episodes_to_go > 0, Fore.CYAN + "Training is already finished" + Fore.RESET

    epochs_time_history = deque(maxlen=self.SHOW_STATS_INTERVAL * 50)

    # Save starting kernels and biases
    if not self.__initiated:
      self.__save_img(save_raw_progress_images)
      self.save_checkpoint()

    print(Fore.GREEN + f"Starting training on episode {self.__episode_counter} for {target_episode} episode" + Fore.RESET)
    print(Fore.MAGENTA + "Preview training stats in tensorboard: http://localhost:6006" + Fore.RESET)
    for _ in range(episodes_to_go):
      ep_start = time.time()

      ### Train Discriminator ###
      # Train discriminator (real as ones and fake as zeros)
      disc_stats = deque(maxlen=discriminator_training_multiplier)

      for _ in range(discriminator_training_multiplier):
        disc_loss, real_loss, fake_loss = self.__train_discriminator(discriminator_smooth_real_labels, discriminator_smooth_fake_labels)
        disc_stats.append([disc_loss, real_loss, fake_loss])

      # Calculate mean of losses of discriminator from all trainings and calculate disc loss
      disc_stats = np.mean(disc_stats, 0)

      if pretrain_episodes and self.__episode_counter < pretrain_episodes:
        ### Pretrain Generator ###
        gen_loss, psnr, psnr_y, ssim = self.__train_generator()
        partial_gan_losses = None
      else:
        ### Train GAN ###
        # Train GAN (wants discriminator to recognize fake images as valid)
        gen_loss, partial_gan_losses, psnr, psnr_y, ssim = self.__train_gan(generator_smooth_labels)

      # Set LR based on episode count and schedule
      # new_gen_lr = self.gen_lr_scheduler.set_lr(self.generator)
      if self.__gen_lr_scheduler.set_lr(self.__combined_generator_model, self.__episode_counter):
        print(Fore.MAGENTA + f"New LR for generator is {self.__gen_lr_scheduler.current_lr}" + Fore.RESET)
      if self.__disc_lr_scheduler.set_lr(self.__discriminator, self.__episode_counter):
        print(Fore.MAGENTA + f"New LR for discriminator is {self.__disc_lr_scheduler.current_lr}" + Fore.RESET)

      # Append stats to stat logger
      self.__stat_logger.append_stats(self.__episode_counter, disc_loss=disc_stats[0], disc_real_loss=disc_stats[2], disc_fake_loss=disc_stats[1], gen_loss=gen_loss, psnr=psnr, psnr_y=psnr_y, ssim=ssim, disc_label_noise=self.__discriminator_label_noise if self.__discriminator_label_noise else 0, gen_lr=self.__gen_lr_scheduler.current_lr, disc_lr=self.__disc_lr_scheduler.current_lr)

      self.__episode_counter += 1
      self.__tensorboard.step = self.__episode_counter

      # Save stats and print them to console
      if self.__episode_counter % self.SHOW_STATS_INTERVAL == 0:
        print(Fore.GREEN + f"{self.__episode_counter}/{target_episode}, Remaining: {(time_to_format(mean(epochs_time_history) * (target_episode - self.__episode_counter))) if epochs_time_history else 'Unable to calculate'}\t\tDiscriminator: [loss: {round(disc_stats[0], 5)}, real_loss: {round(float(disc_stats[2]), 5)}, fake_loss: {round(float(disc_stats[1]), 5)}, label_noise: {round(self.__discriminator_label_noise * 100, 2) if self.__discriminator_label_noise else 0}%] Generator: [loss: {round(gen_loss, 5)}, partial_losses: {partial_gan_losses}, psnr: {round(psnr, 3)}dB, psnr_y: {round(psnr_y, 3)}dB, ssim: {round(ssim, 5)}]\n"
                           f"Generator LR: {self.__gen_lr_scheduler.current_lr}, Discriminator LR: {self.__disc_lr_scheduler.current_lr}" + Fore.RESET)

      # Decay label noise
      if self.__discriminator_label_noise and self.__discriminator_label_noise_decay:
        self.__discriminator_label_noise = max([self.__discriminator_label_noise_min, (self.__discriminator_label_noise * self.__discriminator_label_noise_decay)])

      # Save progress
      if progress_images_save_interval is not None and self.__episode_counter % progress_images_save_interval == 0:
        self.__save_img(save_raw_progress_images)

      # Save weights of models
      if weights_save_interval is not None and self.__episode_counter % weights_save_interval == 0:
        self.__save_weights()

      # Save checkpoint
      if self.__episode_counter % self.CHECKPOINT_SAVE_INTERVAL == 0:
        self.save_checkpoint()
        print(Fore.BLUE + "Checkpoint created" + Fore.RESET)

      # Reset seeds
      if self.__episode_counter % self.RESET_SEEDS_INTERVAL == 0:
        np.random.seed(None)
        random.seed()

      epochs_time_history.append(time.time() - ep_start)

    # Shutdown helper threads
    print(Fore.GREEN + "Training Complete - Waiting for other threads to finish" + Fore.RESET)
    self.__stat_logger.terminate()
    self.batch_maker.terminate()
    self.save_checkpoint()
    self.__save_weights()
    self.batch_maker.join()
    self.__stat_logger.join()
    print(Fore.GREEN + "All threads finished" + Fore.RESET)

  def __save_img(self, save_raw_progress_images:bool=True, tensorflow_description:str="progress"):
    if not os.path.exists(self.__training_progress_save_path + "/progress_images"): os.makedirs(self.__training_progress_save_path + "/progress_images")

    final_image = np.zeros(shape=(self.__target_image_shape[0] * len(self.__progress_test_images_paths), self.__target_image_shape[1] * 3, self.__target_image_shape[2])).astype(np.float32)

    for idx, test_image_path in enumerate(self.__progress_test_images_paths):
      if not os.path.exists(test_image_path):
        print(Fore.YELLOW + f"Failed to locate test image: {test_image_path}, replacing it with new one!" + Fore.RESET)
        self.__progress_test_images_paths[idx] = random.choice(self.__train_data)
        self.save_checkpoint()

      # Load image for upscale and resize it to starting (small) image size
      original_unscaled_image = cv.imread(test_image_path)
      # print(f"[DEBUG] {original_unscaled_image.shape}, {self.target_image_shape}")
      if original_unscaled_image.shape != self.__target_image_shape:
        original_image = cv.resize(original_unscaled_image, dsize=(self.__start_image_shape[1], self.__start_image_shape[0]), interpolation=(cv.INTER_AREA if (original_unscaled_image.shape[0] > self.__start_image_shape[0] and original_unscaled_image.shape[1] > self.__start_image_shape[1]) else cv.INTER_CUBIC))
      else:
        original_image = original_unscaled_image
      small_image = cv.resize(original_image, dsize=(self.__start_image_shape[1], self.__start_image_shape[0]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > self.__start_image_shape[0] and original_image.shape[1] > self.__start_image_shape[1]) else cv.INTER_CUBIC))

      # Conver image to RGB colors and upscale it
      gen_img = self.__generator.predict(np.array([cv.cvtColor(small_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0]))[0]

      # Rescale images 0 to 255
      gen_img = (0.5 * gen_img + 0.5) * 255
      gen_img = cv.cvtColor(gen_img, cv.COLOR_RGB2BGR)

      # Place side by side image resized by opencv, original (large) image and upscaled by gan
      final_image[idx * gen_img.shape[1]:(idx + 1) * gen_img.shape[1], 0:gen_img.shape[0], :] = cv.resize(small_image, dsize=(self.__target_image_shape[1], self.__target_image_shape[0]), interpolation=(cv.INTER_AREA if (small_image.shape[0] > self.__target_image_shape[0] and small_image.shape[1] > self.__target_image_shape[1]) else cv.INTER_CUBIC))
      final_image[idx * gen_img.shape[1]:(idx + 1) * gen_img.shape[1], gen_img.shape[0]:gen_img.shape[0] * 2, :] = original_image
      final_image[idx * gen_img.shape[1]:(idx + 1) * gen_img.shape[1], gen_img.shape[0] * 2:gen_img.shape[0] * 3, :] = gen_img

    # Save image to folder and to tensorboard
    if save_raw_progress_images:
      cv.imwrite(f"{self.__training_progress_save_path}/progress_images/{self.__episode_counter}.png", final_image)
    self.__tensorboard.write_image(np.reshape(cv.cvtColor(final_image, cv.COLOR_BGR2RGB) / 255, (-1, final_image.shape[0], final_image.shape[1], final_image.shape[2])).astype(np.float32), description=tensorflow_description)

  # Save weights of generator and discriminator model
  def __save_weights(self):
    save_dir = self.__training_progress_save_path + "/weights/" + str(self.__episode_counter)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    self.__generator.save_weights(f"{save_dir}/generator_{self.__gen_mod_name}.h5")
    self.__discriminator.save_weights(f"{save_dir}/discriminator_{self.__disc_mod_name}.h5")

  # Load weights to models from given episode
  def load_gen_weights_from_episode(self, episode:int):
    weights_dir = self.__training_progress_save_path + "/weights/" + str(episode)
    if not os.path.exists(weights_dir): return

    gen_weights_path = weights_dir + f"/generator_{self.__gen_mod_name}.h5"
    if os.path.exists(gen_weights_path):
      self.__generator.load_weights(gen_weights_path)

  def load_disc_weights_from_episode(self, episode:int):
    weights_dir = self.__training_progress_save_path + "/weights/" + str(episode)
    if not os.path.exists(weights_dir): return

    disc_weights_path = weights_dir + f"/discriminator_{self.__disc_mod_name}.h5"
    if os.path.exists(disc_weights_path):
      self.__discriminator.load_weights(disc_weights_path)

  # Save images of model structures
  def save_models_structure_images(self):
    save_path = self.__training_progress_save_path + "/model_structures"
    if not os.path.exists(save_path): os.makedirs(save_path)
    plot_model(self.__combined_generator_model, os.path.join(save_path, "combined.png"), expand_nested=True, show_shapes=True)
    plot_model(self.__generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)
    plot_model(self.__discriminator, os.path.join(save_path, "discriminator.png"), expand_nested=True, show_shapes=True)
    plot_model(self.__combined_discriminator_model, os.path.join(save_path, "combined_discriminator.png"), expand_nested=True, show_shapes=True)

  # Load progress of training from checkpoint
  def __load_checkpoint(self):
    checkpoint_base_path = os.path.join(self.__training_progress_save_path, "checkpoint")
    if not os.path.exists(os.path.join(checkpoint_base_path, "checkpoint_data.json")): return

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "rb") as f:
      data = json.load(f)

      if data:
        self.__episode_counter = int(data["episode"])

        try:
          self.__generator.load_weights(data["gen_path"])
        except:
          try:
            self.__generator.load_weights(data["gen_path"] + ".lock")
          except:
            print(Fore.YELLOW + "Failed to load generator weights from checkpoint" + Fore.RESET)

        try:
          self.__discriminator.load_weights(data["disc_path"])
        except:
          try:
            self.__discriminator.load_weights(data["disc_path"] + ".lock")
          except:
            print(Fore.YELLOW + "Failed to load discriminator weights from checkpoint" + Fore.RESET)

        if "disc_label_noise" in data.keys():
          self.__discriminator_label_noise = float(data["disc_label_noise"])

        if not self.__custom_test_images or self.__custom_loading_failed:
          self.__progress_test_images_paths = data["test_image"]
        self.__initiated = True

  # Save progress of training
  def save_checkpoint(self):
    checkpoint_base_path = os.path.join(self.__training_progress_save_path, "checkpoint")
    if not os.path.exists(checkpoint_base_path): os.makedirs(checkpoint_base_path)

    gen_path = f"{checkpoint_base_path}/generator_{self.__gen_mod_name}.h5"
    disc_path = f"{checkpoint_base_path}/discriminator_{self.__disc_mod_name}.h5"

    if os.path.exists(f"{checkpoint_base_path}/generator_{self.__gen_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/generator_{self.__gen_mod_name}.h5.lock")
    if os.path.exists(f"{checkpoint_base_path}/discriminator_{self.__disc_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/discriminator_{self.__disc_mod_name}.h5.lock")

    if os.path.exists(gen_path): os.rename(gen_path, f"{checkpoint_base_path}/generator_{self.__gen_mod_name}.h5.lock")
    if os.path.exists(disc_path): os.rename(disc_path, f"{checkpoint_base_path}/discriminator_{self.__disc_mod_name}.h5.lock")

    self.__generator.save_weights(gen_path)
    self.__discriminator.save_weights(disc_path)

    if os.path.exists(f"{checkpoint_base_path}/generator_{self.__gen_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/generator_{self.__gen_mod_name}.h5.lock")
    if os.path.exists(f"{checkpoint_base_path}/discriminator_{self.__disc_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/discriminator_{self.__disc_mod_name}.h5.lock")

    data = {
      "episode": self.__episode_counter,
      "gen_path": gen_path,
      "disc_path": disc_path,
      "disc_label_noise": self.__discriminator_label_noise,
      "test_image": self.__progress_test_images_paths,
    }

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "w", encoding='utf-8') as f:
      json.dump(data, f)

  def make_progress_gif(self, frame_duration:int=16):
    if not os.path.exists(self.__training_progress_save_path): os.makedirs(self.__training_progress_save_path)
    if not os.path.exists(self.__training_progress_save_path + "/progress_images"): return

    frames = []
    img_file_names = os.listdir(self.__training_progress_save_path + "/progress_images")

    for im_file in img_file_names:
      if os.path.isfile(self.__training_progress_save_path + "/progress_images/" + im_file):
        frames.append(Image.open(self.__training_progress_save_path + "/progress_images/" + im_file))

    if len(frames) > 2:
      frames[0].save(f"{self.__training_progress_save_path}/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, optimize=False, duration=frame_duration, loop=0)