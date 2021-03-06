import os
import numpy as np
from keras.optimizers import Adam, Optimizer
from keras.models import Model
from keras.layers import Input, Dense
from keras.initializers import RandomNormal
from keras.utils import plot_model
from keras.engine.network import Network
import keras.backend as K
from PIL import Image
from cv2 import cv2 as cv
import random
import time
from colorama import Fore
from collections import deque
from typing import Union
import json
from statistics import mean
import imagesize
from multiprocessing.pool import ThreadPool

from ..utils.batch_maker import BatchMaker
from ..models import discriminator_models_spreadsheet, generator_models_spreadsheet
from ..keras_extensions.custom_tensorboard import TensorBoardCustom
from ..utils.helpers import time_to_format, get_paths_of_files_from_path

class DCGAN:
  CONTROL_THRESHOLD = 100_000 # Threshold when after whitch we will be testing training process
  AGREGATE_STAT_INTERVAL = 1_000  # Interval of saving data
  GRADIENT_CHECK_INTERVAL = 10_000  # Interval of checking norm gradient value of combined model
  CHECKPOINT_SAVE_INTERVAL = 1_000  # Interval of saving checkpoint

  def __init__(self, dataset_path:str,
               gen_mod_name: str, disc_mod_name: str,
               latent_dim:int,
               training_progress_save_path:str,
               testing_dataset_path:str=None,
               generator_optimizer:Optimizer=Adam(0.0002, 0.5), discriminator_optimizer:Optimizer=Adam(0.0002, 0.5),
               discriminator_label_noise:float=None, discriminator_label_noise_decay:float=None, discriminator_label_noise_min:float=0.001,
               batch_size: int = 32, buffered_batches:int=20,
               generator_weights:Union[str, None]=None, discriminator_weights:Union[str, None]=None,
               start_episode:int=0, load_from_checkpoint:bool=False,
               check_dataset:bool=True, num_of_loading_workers:int=8):

    self.disc_mod_name = disc_mod_name
    self.gen_mod_name = gen_mod_name
    self.generator_optimizer = generator_optimizer

    self.latent_dim = latent_dim
    assert self.latent_dim > 0, Fore.RED + "Invalid latent dim" + Fore.RESET

    self.batch_size = batch_size
    assert self.batch_size > 0, Fore.RED + "Invalid batch size" + Fore.RESET

    self.discriminator_label_noise = discriminator_label_noise
    self.discriminator_label_noise_decay = discriminator_label_noise_decay
    self.discriminator_label_noise_min = discriminator_label_noise_min

    self.progress_image_dim = (16, 9)

    if start_episode < 0: start_episode = 0
    self.episode_counter = start_episode

    # Initialize training data folder and logging
    self.training_progress_save_path = training_progress_save_path
    self.training_progress_save_path = os.path.join(self.training_progress_save_path, f"{self.gen_mod_name}__{self.disc_mod_name}")
    self.tensorboard = TensorBoardCustom(log_dir=os.path.join(self.training_progress_save_path, "logs"))

    # Create array of input image paths
    self.train_data = get_paths_of_files_from_path(dataset_path, only_files=True)
    assert self.train_data, Fore.RED + "Training dataset is not loaded" + Fore.RESET

    self.testing_data = None
    if testing_dataset_path:
      self.testing_data = get_paths_of_files_from_path(testing_dataset_path)
      assert self.testing_data, Fore.RED + "Testing dataset is not loaded" + Fore.RESET

    # Load one image to get shape of it
    tmp_image = cv.imread(self.train_data[0])
    self.image_shape = tmp_image.shape
    self.image_channels = self.image_shape[2]

    # Check image size validity
    if self.image_shape[0] < 4 or self.image_shape[1] < 4: raise Exception("Images too small, min size (4, 4)")

    # Check validity of whole datasets
    if check_dataset:
      self.validate_dataset()

    # Define static vars
    if os.path.exists(f"{self.training_progress_save_path}/static_noise.npy"):
      self.static_noise = np.load(f"{self.training_progress_save_path}/static_noise.npy")
      if self.static_noise.shape[0] != (self.progress_image_dim[0] * self.progress_image_dim[1]):
        print(Fore.YELLOW + "Progress image dim changed, restarting static noise!" + Fore.RESET)
        os.remove(f"{self.training_progress_save_path}/static_noise.npy")
        self.static_noise = np.random.normal(0.0, 1.0, size=(self.progress_image_dim[0] * self.progress_image_dim[1], self.latent_dim))
    else:
      self.static_noise = np.random.normal(0.0, 1.0, size=(self.progress_image_dim[0] * self.progress_image_dim[1], self.latent_dim))
    self.kernel_initializer = RandomNormal(stddev=0.02)

    # Load checkpoint
    self.initiated = False
    loaded_gen_weights_path = None
    loaded_disc_weights_path = None
    if load_from_checkpoint:
      loaded_gen_weights_path, loaded_disc_weights_path = self.load_checkpoint()

    # Create batchmaker and start it
    self.batch_maker = BatchMaker(self.train_data, self.batch_size, buffered_batches=buffered_batches, num_of_loading_workers=num_of_loading_workers)

    self.testing_batchmaker = None
    if self.testing_data:
      self.testing_batchmaker = BatchMaker(self.testing_data, self.batch_size, buffered_batches=buffered_batches, num_of_loading_workers=num_of_loading_workers)
      self.testing_batchmaker.start()

    #################################
    ###   Create discriminator    ###
    #################################
    self.discriminator = self.build_discriminator(disc_mod_name)
    self.discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer)

    #################################
    ###     Create generator      ###
    #################################
    self.generator = self.build_generator(gen_mod_name)
    if self.generator.output_shape[1:] != self.image_shape: raise Exception("Invalid image input size for this generator model")

    #################################
    ### Create combined generator ###
    #################################
    noise_input = Input(shape=(self.latent_dim,), name="noise_input")
    gen_images = self.generator(noise_input)

    # Create frozen version of discriminator
    frozen_discriminator = Network(self.discriminator.inputs, self.discriminator.outputs, name="frozen_discriminator")
    frozen_discriminator.trainable = False

    # Discriminator takes images and determinates validity
    valid = frozen_discriminator(gen_images)

    # Combine models
    # Train generator to fool discriminator
    self.combined_generator_model = Model(noise_input, valid, name="dcgan_model")
    self.combined_generator_model.compile(loss="binary_crossentropy", optimizer=self.generator_optimizer)

    # Print all summaries
    print("\nDiscriminator Summary:")
    self.discriminator.summary()
    print("\nGenerator Summary:")
    self.generator.summary()
    print("\nGAN Summary")
    self.combined_generator_model.summary()

    # Load weights from checkpoint
    try:
      if loaded_gen_weights_path: self.generator.load_weights(loaded_gen_weights_path)
    except:
      print(Fore.YELLOW + "Failed to load generator weights from checkpoint" + Fore.RESET)

    try:
      if loaded_disc_weights_path: self.discriminator.load_weights(loaded_disc_weights_path)
    except:
      print(Fore.YELLOW + "Failed to load discriminator weights from checkpoint" + Fore.RESET)

    # Load weights from param and override checkpoint weights
    if generator_weights: self.generator.load_weights(generator_weights)
    if discriminator_weights: self.discriminator.load_weights(discriminator_weights)

  # Function for creating gradient generator
  def gradient_norm_generator(self):
    grads = K.gradients(self.combined_generator_model.total_loss, self.combined_generator_model.trainable_weights)
    summed_squares = [K.sum(K.square(g)) for g in grads]
    norm = K.sqrt(sum(summed_squares))
    inputs = self.combined_generator_model._feed_inputs + self.combined_generator_model._feed_targets + self.combined_generator_model._feed_sample_weights
    func = K.function(inputs, [norm])
    return func

  # Check if datasets have consistent shapes
  def validate_dataset(self):
    def check_image(image_path):
      im_shape = imagesize.get(image_path)
      if im_shape[0] != self.image_shape[0] or im_shape[1] != self.image_shape[1]:
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
  def build_generator(self, model_name:str):
    noise = Input(shape=(self.latent_dim,))

    try:
      m = getattr(generator_models_spreadsheet, model_name)(noise, self.image_shape, self.image_channels, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Generator model not found!\n{e}")

    return Model(noise, m, name="generator_model")

  # Create discriminator based on teplate selected by name
  def build_discriminator(self, model_name:str, classification:bool=True):
    img = Input(shape=self.image_shape)

    try:
      m = getattr(discriminator_models_spreadsheet, model_name)(img, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Discriminator model not found!\n{e}")

    if classification:
      m = Dense(1, activation="sigmoid")(m)

    return Model(img, m, name="discriminator_model")

  def train(self, target_episode:int,
            feed_prev_gen_batch:bool=False, feed_old_perc_amount:float=0.2,
            progress_images_save_interval:int=None, save_raw_progress_images:bool=True, weights_save_interval:int=None,
            discriminator_smooth_real_labels:bool=False, discriminator_smooth_fake_labels:bool=False,
            generator_smooth_labels:bool=False):

    # Function for adding random noise to labels (flipping them)
    def noising_labels(labels: np.ndarray, noise_ammount:float=0.01):
      array = np.zeros(labels.shape)
      for idx in range(labels.shape[0]):
        if random.random() < noise_ammount:
          array[idx] = 1 - labels[idx]
          if array[idx] < 0: array[idx] = -array[idx]
        else:
          array[idx] = labels[idx]
      return labels

    # Function for replacing new generated images with old generated images
    def replace_random_images(orig_images: np.ndarray, repl_images: deque, perc_ammount:float=0.20):
      repl_images = np.array(repl_images)
      for idx in range(orig_images.shape[0]):
        if random.random() < perc_ammount:
          orig_images[idx] = repl_images[random.randint(0, repl_images.shape[0] - 1)]
      return orig_images

    # Check arguments and input data
    assert target_episode > 0, Fore.RED + "Invalid number of epochs" + Fore.RESET
    if progress_images_save_interval:
      assert progress_images_save_interval <= target_episode, Fore.RED + "Invalid progress save interval" + Fore.RESET
    if weights_save_interval:
      assert weights_save_interval <= target_episode, Fore.RED + "Invalid weights save interval" + Fore.RESET

    # Calculate epochs to go
    end_episode = target_episode
    target_episode = target_episode - self.episode_counter
    assert target_episode > 0, Fore.CYAN + "Training is already finished" + Fore.RESET

    # Save noise for progress consistency
    if progress_images_save_interval is not None:
      if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)
      np.save(f"{self.training_progress_save_path}/static_noise.npy", self.static_noise)

    # Training variables
    prev_gen_images = deque(maxlen=3*self.batch_size)
    get_gradients = self.gradient_norm_generator()

    epochs_time_history = deque(maxlen=self.AGREGATE_STAT_INTERVAL * 50)

    # Save starting kernels and biases
    if not self.initiated:
      self.__save_imgs(save_raw_progress_images)
      self.tensorboard.log_kernels_and_biases(self.generator)
      self.save_checkpoint()

    print(Fore.GREEN + f"Starting training on episode {self.episode_counter} for {target_episode} episodes" + Fore.RESET)
    for _ in range(target_episode):
      ep_start = time.time()

      ### Train Discriminator ###
      # Select batch of valid images
      imgs = self.batch_maker.get_batch()

      # Sample noise and generate new images
      gen_imgs = self.generator.predict(np.random.normal(0.0, 1.0, (self.batch_size, self.latent_dim)))

      # Train discriminator (real as ones and fake as zeros)
      if discriminator_smooth_real_labels:
        disc_real_labels = np.random.uniform(0.8, 1.0, size=(self.batch_size, 1))
      else:
        disc_real_labels = np.ones(shape=(self.batch_size, 1))

      if discriminator_smooth_fake_labels:
        disc_fake_labels = np.random.uniform(0, 0.2, size=(self.batch_size, 1))
      else:
        disc_fake_labels = np.zeros(shape=(self.batch_size, 1))

      if feed_prev_gen_batch:
        if len(prev_gen_images) > 0:
          tmp_imgs = replace_random_images(gen_imgs, prev_gen_images, feed_old_perc_amount)
          prev_gen_images += deque(gen_imgs)
          gen_imgs = tmp_imgs
        else:
          prev_gen_images += deque(gen_imgs)

      # Adding random noise to discriminator labels
      if self.discriminator_label_noise and self.discriminator_label_noise > 0:
        disc_real_labels = noising_labels(disc_real_labels, self.discriminator_label_noise / 2)
        disc_fake_labels = noising_labels(disc_fake_labels, self.discriminator_label_noise / 2)

      self.discriminator.trainable = True
      disc_real_loss = self.discriminator.train_on_batch(imgs, disc_real_labels)
      disc_fake_loss = self.discriminator.train_on_batch(gen_imgs, disc_fake_labels)

      ### Train Generator ###
      # Train generator (wants discriminator to recognize fake images as valid)
      if generator_smooth_labels:
        gen_labels = np.random.uniform(0.8, 1.0, size=(self.batch_size, 1))
      else:
        gen_labels = np.ones(shape=(self.batch_size, 1))
      self.discriminator.trainable = False
      gan_loss = self.combined_generator_model.train_on_batch(np.random.normal(0.0, 1.0, (self.batch_size, self.latent_dim)), gen_labels)

      self.episode_counter += 1
      self.tensorboard.step = self.episode_counter
      self.tensorboard.update_stats(disc_real_loss=disc_real_loss, disc_fake_loss=disc_fake_loss, gan_loss=gan_loss, disc_label_noise=self.discriminator_label_noise if self.discriminator_label_noise else 0)

      # Decay label noise
      if self.discriminator_label_noise and self.discriminator_label_noise_decay:
        self.discriminator_label_noise = max([self.discriminator_label_noise_min, (self.discriminator_label_noise * self.discriminator_label_noise_decay)])

        if (self.discriminator_label_noise_min == 0) and (self.discriminator_label_noise != 0) and (self.discriminator_label_noise < 0.001):
          self.discriminator_label_noise = 0

      # Seve stats and print them to console
      if self.episode_counter % self.AGREGATE_STAT_INTERVAL == 0:
        self.tensorboard.log_kernels_and_biases(self.generator)

        # Change color of log according to state of training
        print(Fore.GREEN + f"{self.episode_counter}/{end_episode}, Remaining: {time_to_format(mean(epochs_time_history) * (end_episode - self.episode_counter))}\t\t[D-R loss: {round(float(disc_real_loss), 5)}, D-F loss: {round(float(disc_fake_loss), 5)}] [G loss: {round(float(gan_loss), 5)}] - Epsilon: {round(self.discriminator_label_noise, 4) if self.discriminator_label_noise else 0}" + Fore.RESET)

      # Save progress
      if self.training_progress_save_path is not None and progress_images_save_interval is not None and self.episode_counter % progress_images_save_interval == 0:
        self.__save_imgs(save_raw_progress_images)

      # Save weights of models
      if weights_save_interval is not None and self.episode_counter % weights_save_interval == 0:
        self.__save_weights()

      # Save checkpoint
      if self.episode_counter % self.CHECKPOINT_SAVE_INTERVAL == 0:
        self.save_checkpoint()
        print(Fore.BLUE + "Checkpoint created" + Fore.RESET)

      if self.episode_counter % self.GRADIENT_CHECK_INTERVAL == 0:
        # Generate evaluation noise and labels
        eval_noise = np.random.normal(0.0, 1.0, (self.batch_size, self.latent_dim))
        eval_labels = np.ones(shape=(self.batch_size, 1))

        # Create gradient function and evaluate based on eval noise and labels
        norm_gradient = get_gradients([eval_noise, eval_labels, np.ones(len(eval_labels))])[0]

        # Check norm gradient
        if norm_gradient > 100 and self.episode_counter > self.CONTROL_THRESHOLD:
          print(Fore.RED + f"Current generator norm gradient: {norm_gradient}")
          print("Gradient too high!" + Fore.RESET)
          if input("Do you want exit training?\n") == "y": return
        elif norm_gradient < 0.2 and self.episode_counter > self.CONTROL_THRESHOLD:
          print(Fore.RED + f"Current generator norm gradient: {norm_gradient}")
          print("Gradient vanished!" + Fore.RESET)
          if input("Do you want exit training?\n") == "y": return
        else:
          print(Fore.BLUE + f"Current generator norm gradient: {norm_gradient}" + Fore.RESET)

        # Change seed
        np.random.seed(None)
        random.seed()

      epochs_time_history.append(time.time() - ep_start)

    # Shutdown helper threads
    print(Fore.GREEN + "Training Complete - Waiting for other threads to finish" + Fore.RESET)
    if self.testing_batchmaker: self.testing_batchmaker.__terminate = True
    self.batch_maker.terminate()
    self.save_checkpoint()
    self.__save_weights()
    self.batch_maker.join()
    if self.testing_batchmaker: self.testing_batchmaker.join()
    print(Fore.GREEN + "All threads finished" + Fore.RESET)

  # Function for saving progress images
  def __save_imgs(self, save_raw_progress_images:bool=True):
    if not os.path.exists(self.training_progress_save_path + "/progress_images"): os.makedirs(self.training_progress_save_path + "/progress_images")
    gen_imgs = self.generator.predict(self.static_noise)

    # Rescale images 0 to 255
    gen_imgs = (0.5 * gen_imgs + 0.5) * 255

    final_image = np.zeros(shape=(self.image_shape[0] * self.progress_image_dim[1], self.image_shape[1] * self.progress_image_dim[0], self.image_channels)).astype(np.float32)

    cnt = 0
    for i in range(self.progress_image_dim[1]):
      for j in range(self.progress_image_dim[0]):
        if self.image_channels == 3:
          final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[1] * j:self.image_shape[1] * (j + 1), :] = gen_imgs[cnt]
        else:
          final_image[self.image_shape[0] * i:self.image_shape[0] * (i + 1), self.image_shape[1] * j:self.image_shape[1] * (j + 1), 0] = gen_imgs[cnt, :, :, 0]
        cnt += 1
    final_image = cv.cvtColor(final_image, cv.COLOR_RGB2BGR)

    if save_raw_progress_images:
      cv.imwrite(f"{self.training_progress_save_path}/progress_images/{self.episode_counter}.png", final_image)
    self.tensorboard.write_image(np.reshape(cv.cvtColor(final_image, cv.COLOR_BGR2RGB) / 255, (-1, final_image.shape[0], final_image.shape[1], final_image.shape[2])).astype(np.float32))

  def save_models_structure_images(self):
    save_path = self.training_progress_save_path + "/model_structures"
    if not os.path.exists(save_path): os.makedirs(save_path)
    plot_model(self.combined_generator_model, os.path.join(save_path, "combined.png"), expand_nested=True, show_shapes=True)
    plot_model(self.generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)
    plot_model(self.discriminator, os.path.join(save_path, "discriminator.png"), expand_nested=True, show_shapes=True)

  def load_checkpoint(self):
    checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
    if not os.path.exists(os.path.join(checkpoint_base_path, "checkpoint_data.json")): return None, None

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "rb") as f:
      data = json.load(f)

      if data:
        self.episode_counter = int(data["episode"])
        if data["disc_label_noise"]:
          self.discriminator_label_noise = float(data["disc_label_noise"])
        self.initiated = True
        return data["gen_path"], data["disc_path"]
      return None, None

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
      "disc_label_noise": self.discriminator_label_noise
    }

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "w", encoding='utf-8') as f:
      json.dump(data, f)

  def __save_weights(self):
    save_dir = self.training_progress_save_path + "/weights/" + str(self.episode_counter)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")
    self.discriminator.save_weights(f"{save_dir}/discriminator_{self.disc_mod_name}.h5")

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