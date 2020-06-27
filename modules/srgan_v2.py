from colorama import Fore
import os
from typing import Union
import tensorflow as tf
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
from tqdm import tqdm
import json
import random
import time
import imagesize
from multiprocessing.pool import ThreadPool

from modules.models import upscaling_generator_models_spreadsheet, discriminator_models_spreadsheet
from modules.custom_tensorboard import TensorBoardCustom
from modules.batch_maker import BatchMaker
from modules.helpers import time_to_format

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
  # Input image to extract features from
  img = Input(shape=image_shape)

  # Get the vgg network. Extract features from last conv layer
  vgg = VGG19(weights="imagenet")
  vgg.trainable = False
  for l in vgg.layers:
    l.trainable = False

  vgg.outputs = [vgg.layers[20].output]

  # Create model and compile
  model = Model(inputs=img, outputs=vgg(img))
  model.trainable = False
  return model

class SRGAN_V2:
  AGREGATE_STAT_INTERVAL = 1  # Interval of saving data
  RESET_SEEDS_INTERVAL = 10  # Interval of checking norm gradient value of combined model
  CHECKPOINT_SAVE_INTERVAL = 1  # Interval of saving checkpoint

  def __init__(self, dataset_path:str, num_of_upscales:int,
               gen_mod_name: str, disc_mod_name: str,
               training_progress_save_path:str,
               generator_optimizer: Optimizer = Adam(0.0001, 0.9), discriminator_optimizer: Optimizer = Adam(0.0001, 0.9),
               discriminator_label_noise: float = None, discriminator_label_noise_decay: float = None, discriminator_label_noise_min: float = 0.001,
               batch_size: int = 32, buffered_batches:int=20, test_batches:int=1,
               generator_weights: Union[str, None, int] = None, discriminator_weights: Union[str, None, int] = None,
               start_episode: int = 0, load_from_checkpoint: bool = False,
               custom_batches_per_epochs:int=None, custom_hr_test_image_path:str=None, check_dataset:bool=True):

    self.disc_mod_name = disc_mod_name
    self.gen_mod_name = gen_mod_name
    self.num_of_upscales = num_of_upscales
    assert self.num_of_upscales >= 0, Fore.RED + "Invalid number of upscales" + Fore.RESET

    self.discriminator_label_noise = discriminator_label_noise
    self.discriminator_label_noise_decay = discriminator_label_noise_decay
    self.discriminator_label_noise_min = discriminator_label_noise_min

    self.batch_size = batch_size
    assert self.batch_size > 0, Fore.RED + "Invalid batch size" + Fore.RESET

    self.test_batches = test_batches
    assert self.test_batches > 0, Fore.RED + "Invalid test batch size" + Fore.RESET

    if start_episode < 0: start_episode = 0
    self.epoch_counter = start_episode
    if custom_batches_per_epochs and custom_batches_per_epochs < 1: custom_batches_per_epochs = None
    self.custom_batches_per_epochs = custom_batches_per_epochs

    # Create array of input image paths
    self.train_data = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    assert self.train_data, Fore.RED + "Dataset is not loaded" + Fore.RESET
    self.data_length = len(self.train_data)
    assert self.data_length > 0, Fore.RED + "Dataset is not loaded" + Fore.RESET

    # Load one image to get shape of it
    self.target_image_shape = cv.imread(self.train_data[0]).shape

    # Check image size validity
    if self.target_image_shape[0] < 4 or self.target_image_shape[1] < 4: raise Exception("Images too small, min size (4, 4)")

    # Starting image size calculate
    self.start_image_shape = count_upscaling_start_size(self.target_image_shape, self.num_of_upscales)

    # Check validity of whole datasets
    if check_dataset:
      self.validate_dataset()

    # Initialize training data folder and logging
    self.training_progress_save_path = training_progress_save_path
    self.training_progress_save_path = os.path.join(self.training_progress_save_path, f"{self.gen_mod_name}__{self.disc_mod_name}__{self.start_image_shape}_to_{self.target_image_shape}")
    self.tensorboard = TensorBoardCustom(log_dir=os.path.join(self.training_progress_save_path, "logs"))

    # Create array of input image paths
    self.train_data = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    self.data_length = len(self.train_data)

    # Define static vars
    self.kernel_initializer = RandomNormal(stddev=0.02)
    self.custom_hr_test_image_path = custom_hr_test_image_path
    if custom_hr_test_image_path and os.path.exists(custom_hr_test_image_path):
      self.progress_test_image_path = custom_hr_test_image_path
    else:
      self.progress_test_image_path = random.choice(self.train_data)

    # Create batchmaker and start it
    self.batch_maker = BatchMaker(self.train_data, self.data_length, self.batch_size, buffered_batches=buffered_batches, secondary_size=self.start_image_shape)
    self.batch_maker.start()

    #################################
    ###   Create discriminator    ###
    #################################
    self.discriminator = self.build_discriminator(disc_mod_name)
    self.discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer, metrics=["accuracy"])
    print("\nDiscriminator Sumary:")
    self.discriminator.summary()

    #################################
    ###     Create generator      ###
    #################################
    self.generator = self.build_generator(gen_mod_name)
    if self.generator.output_shape[1:] != self.target_image_shape: raise Exception("Invalid image input size for this generator model")
    self.generator.compile(loss="mse", optimizer=generator_optimizer)
    print("\nGenerator Sumary:")
    self.generator.summary()

    #################################
    ###    Create vgg network     ###
    #################################
    self.vgg = build_vgg(self.target_image_shape)
    self.vgg.compile(loss='mse', optimizer=Adam(0.0001, 0.9), metrics=['accuracy'])

    #################################
    ### Create combined generator ###
    #################################
    small_image_input = Input(shape=self.start_image_shape, name="small_image_input")
    gen_images = self.generator(small_image_input)
    generated_features = self.vgg(preprocess_vgg(gen_images))

    # Create frozen version of discriminator
    frozen_discriminator = Network(self.discriminator.inputs, self.discriminator.outputs, name="frozen_discriminator")
    frozen_discriminator.trainable = False

    # Discriminator takes images and determinates validity
    validity = frozen_discriminator(gen_images)

    # Combine models
    # Train generator to fool discriminator
    self.combined_generator_model = Model(small_image_input, outputs=[generated_features, validity], name="srgan_model")
    self.combined_generator_model.compile(loss=["mse", "binary_crossentropy"],
                                          loss_weights=[0.006, 1e-3],
                                          optimizer=generator_optimizer)

    # Load checkpoint
    self.initiated = False
    if load_from_checkpoint: self.load_checkpoint()

    # Load weights from param and override checkpoint weights
    if generator_weights: self.generator.load_weights(generator_weights)
    if discriminator_weights: self.discriminator.load_weights(discriminator_weights)

  # Check if datasets have consistent shapes
  def validate_dataset(self):
    def check_image(image_path):
      im_shape = imagesize.get(image_path)
      if im_shape[0] != self.target_image_shape[0] or im_shape[1] != self.target_image_shape[1]:
        return False
      return True

    print(Fore.BLUE + "Checking dataset validity" + Fore.RESET)
    with ThreadPool(processes=8) as p:
      res = p.map(check_image, self.train_data)
      if not all(res): raise Exception("Inconsistent dataset")

    print(Fore.BLUE + "Dataset valid" + Fore.RESET)

  # Create generator based on template selected by name
  def build_generator(self, model_name:str):
    small_image_input = Input(shape=self.start_image_shape)

    try:
      m = getattr(upscaling_generator_models_spreadsheet, model_name)(small_image_input, self.start_image_shape, self.num_of_upscales, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Generator model not found!\n{e}")

    return Model(small_image_input, m, name="generator_model")

  # Create discriminator based on teplate selected by name
  def build_discriminator(self, model_name:str, classification:bool=True):
    img = Input(shape=self.target_image_shape)

    try:
      m = getattr(discriminator_models_spreadsheet, model_name)(img, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Discriminator model not found!\n{e}")

    if classification:
      m = Dense(1, activation="sigmoid")(m)

    return Model(img, m, name="discriminator_model")

  def train(self, epochs: int, pretrain_epochs:int=None,
            progress_images_save_interval: int = None, save_raw_progress_images:bool=True, weights_save_interval:int=None):

    # Check arguments and input data
    assert epochs > 0, Fore.RED + "Invalid number of epochs" + Fore.RESET
    if progress_images_save_interval is not None and progress_images_save_interval <= epochs and epochs % progress_images_save_interval != 0: raise Exception("Invalid progress save interval")
    if weights_save_interval is not None and weights_save_interval <= epochs and epochs % weights_save_interval != 0: raise Exception("Invalid weights save interval")

    if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)

    # Training variables
    num_of_batches = self.data_length // self.batch_size
    if self.custom_batches_per_epochs: num_of_batches = self.custom_batches_per_epochs
    end_epoch = self.epoch_counter + epochs

    epochs_time_history = deque(maxlen=5)

    # Save starting kernels and biases
    pretrain_active = False
    if not self.initiated:
      self.__save_img(save_raw_progress_images)
      self.tensorboard.log_kernels_and_biases(self.generator)
      if pretrain_epochs:
        assert pretrain_epochs > 0, Fore.RED + "Invalid pretrain epochs" + Fore.RESET
        print(Fore.BLUE + "Pretrain active" + Fore.RESET)
        epochs += pretrain_epochs
        end_epoch += pretrain_epochs
        pretrain_active = True
      self.save_checkpoint()

    print(Fore.GREEN + f"Starting training on epoch {self.epoch_counter} for {epochs} epochs" + Fore.RESET)
    for _ in range(epochs):
      ep_start = time.time()
      for _ in tqdm(range(num_of_batches), unit="batches", smoothing=0.5, leave=False):
        if pretrain_active and self.epoch_counter < pretrain_epochs:
          # Pretrain generator
          large_images, small_images = self.batch_maker.get_batch()
          self.generator.train_on_batch(small_images, large_images)
          time.sleep(0.05)
          continue

        large_images, small_images = self.batch_maker.get_batch()

        gen_imgs = self.generator.predict(small_images)

        # Train discriminator (real as ones and fake as zeros)
        disc_real_labels = np.ones(shape=(self.batch_size, 1))
        disc_fake_labels = np.zeros(shape=(self.batch_size, 1))

        self.discriminator.train_on_batch(large_images, disc_real_labels)
        self.discriminator.train_on_batch(gen_imgs, disc_fake_labels)

        ### Train Generator ###
        # Train generator (wants discriminator to recognize fake images as valid)
        large_images, small_images = self.batch_maker.get_batch()
        gen_labels = np.ones(shape=(self.batch_size, 1))
        predicted_features = self.vgg.predict(preprocess_vgg(large_images))
        self.combined_generator_model.train_on_batch(small_images, [predicted_features, gen_labels])
        time.sleep(0.05)

      time.sleep(0.5)
      self.epoch_counter += 1
      epochs_time_history.append(time.time() - ep_start)
      self.tensorboard.step = self.epoch_counter

      # Decay label noise
      if self.discriminator_label_noise and self.discriminator_label_noise_decay:
        self.discriminator_label_noise = max([self.discriminator_label_noise_min, (self.discriminator_label_noise * self.discriminator_label_noise_decay)])

        if (self.discriminator_label_noise_min == 0) and (self.discriminator_label_noise != 0) and (self.discriminator_label_noise < 0.0001):
          self.discriminator_label_noise = 0

      # Seve stats and print them to console
      if self.epoch_counter % self.AGREGATE_STAT_INTERVAL == 0:
        gen_loss = 0
        mse_gen_loss = 0
        binary_gen_loss = 0
        disc_real_loss = 0
        disc_fake_loss = 0
        disc_real_acc = 0
        disc_fake_acc = 0
        for _ in range(self.test_batches):
          large_images, small_images = self.batch_maker.get_batch()
          gen_imgs = self.generator.predict(small_images)

          # Evaluate models state
          d_r_l, d_r_a = self.discriminator.test_on_batch(large_images, np.ones(shape=(large_images.shape[0], 1)))
          d_f_l, d_f_a = self.discriminator.test_on_batch(gen_imgs, np.zeros(shape=(gen_imgs.shape[0], 1)))
          predicted_features = self.vgg.predict(preprocess_vgg(large_images))
          g_l = self.combined_generator_model.test_on_batch(small_images, [predicted_features, np.ones(shape=(large_images.shape[0], 1))])

          gen_loss += g_l[0]
          mse_gen_loss += g_l[1]
          binary_gen_loss += g_l[2]
          disc_real_loss += d_r_l
          disc_fake_loss += d_f_l
          disc_real_acc += d_r_a
          disc_fake_acc += d_f_a

        # Calculate excatc values of stats and convert accuracy to percents
        gen_loss /= self.test_batches
        mse_gen_loss /= self.test_batches
        binary_gen_loss /= self.test_batches
        disc_real_loss /= self.test_batches
        disc_fake_loss /= self.test_batches
        disc_real_acc /= self.test_batches
        disc_fake_acc /= self.test_batches
        disc_real_acc *= 100.0
        disc_fake_acc *= 100.0

        self.tensorboard.log_kernels_and_biases(self.generator)
        self.tensorboard.update_stats(self.epoch_counter, disc_real_loss=disc_real_loss, disc_real_acc=disc_real_acc, disc_fake_loss=disc_fake_loss, disc_fake_acc=disc_fake_acc, gen_loss=gen_loss, mse_gen_loss=mse_gen_loss, gen_binary_loss=binary_gen_loss, disc_label_noise=self.discriminator_label_noise if self.discriminator_label_noise else 0)

        print(Fore.GREEN + f"{self.epoch_counter}/{end_epoch}, Remaining: {time_to_format(mean(epochs_time_history) * (end_epoch - self.epoch_counter))} - [D-R loss: {round(float(disc_real_loss), 5)}, D-R acc: {round(float(disc_real_acc), 2)}%, D-F loss: {round(float(disc_fake_loss), 5)}, D-F acc: {round(float(disc_fake_acc), 2)}%] [G loss: {round(float(gen_loss), 5)}, G mse_loss: {round(float(mse_gen_loss), 5)}, G binary_loss: {round(float(binary_gen_loss), 5)}] - Epsilon: {round(self.discriminator_label_noise, 4) if self.discriminator_label_noise else 0}" + Fore.RESET)

      # Save progress
      if self.training_progress_save_path is not None and progress_images_save_interval is not None and self.epoch_counter % progress_images_save_interval == 0:
        self.__save_img(save_raw_progress_images)

      # Save weights of models
      if weights_save_interval is not None and self.epoch_counter % weights_save_interval == 0:
        self.save_weights()

      # Save checkpoint
      if self.epoch_counter % self.CHECKPOINT_SAVE_INTERVAL == 0:
        self.save_checkpoint()
        print(Fore.BLUE + "Checkpoint created" + Fore.RESET)

      # Reset seeds
      if self.epoch_counter % self.RESET_SEEDS_INTERVAL == 0:
        np.random.seed(None)
        random.seed()

    # Shutdown helper threads
    print(Fore.GREEN + "Training Complete - Waiting for other threads to finish" + Fore.RESET)
    self.batch_maker.terminate = True
    self.save_checkpoint()
    self.save_weights()
    self.batch_maker.join()
    print(Fore.GREEN + "All threads finished" + Fore.RESET)

  def __save_img(self, save_raw_progress_images:bool=True):
    if not os.path.exists(self.training_progress_save_path + "/progress_images"): os.makedirs(self.training_progress_save_path + "/progress_images")
    if not os.path.exists(self.progress_test_image_path):
      print(Fore.YELLOW + "Test image doesnt exist anymore, choosing new one" + Fore.RESET)
      self.progress_test_image_path = random.choice(self.train_data)
      self.save_checkpoint()

    original_image = cv.imread(self.progress_test_image_path)
    small_image = cv.resize(original_image, dsize=(self.start_image_shape[0], self.start_image_shape[1]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > self.start_image_shape[0] and original_image.shape[1] > self.start_image_shape[1]) else cv.INTER_CUBIC))
    gen_img = self.generator.predict(np.array([cv.cvtColor(small_image, cv.COLOR_BGR2RGB) / 127.5 - 1.0]))[0]

    # Rescale images 0 to 255
    gen_img = (0.5 * gen_img + 0.5) * 255
    gen_img = cv.cvtColor(gen_img, cv.COLOR_RGB2BGR)

    final_image = np.zeros(shape=(gen_img.shape[0], gen_img.shape[1] * 3, gen_img.shape[2])).astype(np.float32)
    final_image[:, 0:gen_img.shape[0], :] = cv.resize(small_image, dsize=(self.target_image_shape[0], self.target_image_shape[1]), interpolation=(cv.INTER_AREA if (small_image.shape[0] > self.target_image_shape[0] and small_image.shape[1] > self.target_image_shape[1]) else cv.INTER_CUBIC))
    final_image[:, gen_img.shape[0]:gen_img.shape[0] * 2, :] = original_image
    final_image[:, gen_img.shape[0] * 2:gen_img.shape[0] * 3, :] = gen_img

    if save_raw_progress_images:
      cv.imwrite(f"{self.training_progress_save_path}/progress_images/{self.epoch_counter}.png", final_image)
    self.tensorboard.write_image(np.reshape(cv.cvtColor(final_image, cv.COLOR_BGR2RGB) / 255, (-1, final_image.shape[0], final_image.shape[1], final_image.shape[2])).astype(np.float32))

  def save_weights(self):
    save_dir = self.training_progress_save_path + "/weights/" + str(self.epoch_counter)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")
    self.discriminator.save_weights(f"{save_dir}/discriminator_{self.disc_mod_name}.h5")

  def save_models_structure_images(self):
    save_path = self.training_progress_save_path + "/model_structures"
    if not os.path.exists(save_path): os.makedirs(save_path)
    plot_model(self.combined_generator_model, os.path.join(save_path, "combined.png"), expand_nested=True, show_shapes=True)
    plot_model(self.generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)
    plot_model(self.discriminator, os.path.join(save_path, "discriminator.png"), expand_nested=True, show_shapes=True)

  def load_checkpoint(self):
    checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
    if not os.path.exists(os.path.join(checkpoint_base_path, "checkpoint_data.json")): return

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "rb") as f:
      data = json.load(f)

      if data:
        self.epoch_counter = int(data["episode"])

        try:
          self.generator.load_weights(data["gen_path"])
        except:
          print(Fore.YELLOW + "Failed to load generator weights from checkpoint" + Fore.RESET)

        try:
          self.discriminator.load_weights(data["disc_path"])
        except:
          print(Fore.YELLOW + "Failed to load discriminator weights from checkpoint" + Fore.RESET)

        if data["disc_label_noise"]:
          self.discriminator_label_noise = float(data["disc_label_noise"])
        if not self.custom_hr_test_image_path:
          self.progress_test_image_path = data["test_image"]
        self.initiated = True

  def save_checkpoint(self):
    checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
    if not os.path.exists(checkpoint_base_path): os.makedirs(checkpoint_base_path)

    self.generator.save_weights(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5")
    self.discriminator.save_weights(f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5")

    data = {
      "episode": self.epoch_counter,
      "gen_path": f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5",
      "disc_path": f"{checkpoint_base_path}/discriminator_{self.disc_mod_name}.h5",
      "disc_label_noise": self.discriminator_label_noise,
      "test_image": self.progress_test_image_path
    }

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "w", encoding='utf-8') as f:
      json.dump(data, f)

  def make_progress_gif(self, frame_duration:int=16):
    if not os.path.exists(self.training_progress_save_path + "/progress_images"): return
    if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)

    frames = []
    img_file_names = os.listdir(self.training_progress_save_path + "/progress_images")

    for im_file in img_file_names:
      if os.path.isfile(self.training_progress_save_path + "/progress_images/" + im_file):
        frames.append(Image.open(self.training_progress_save_path + "/progress_images/" + im_file))

    if len(frames) > 2:
      frames[0].save(f"{self.training_progress_save_path}/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, optimize=False, duration=frame_duration, loop=0)