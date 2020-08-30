import json
import os
import numpy as np
from keras.optimizers import RMSprop, Optimizer
from keras.models import Model
from keras.layers import Input, Dense
from keras.initializers import RandomNormal
from keras.utils import plot_model
from keras.layers import Layer
import keras.backend as K
from PIL import Image
import cv2 as cv
import random
import time
from colorama import Fore
from functools import partial
from typing import Union
from collections import deque
from statistics import mean
import imagesize
from multiprocessing.pool import ThreadPool

from modules.batch_maker import BatchMaker
from modules.models import discriminator_models_spreadsheet, generator_models_spreadsheet
from modules.keras_extensions.custom_tensorboard import TensorBoardCustom
from modules.helpers import time_to_format, get_paths_of_files_from_path

# Custom loss function
def wasserstein_loss(y_true, y_pred):
  return K.mean(y_true * y_pred)

# Gradient penalty loss function
def gradient_penalty_loss(_, y_pred, averaged_samples):
  gradients = K.gradients(y_pred, averaged_samples)[0]
  gradients_sqr = K.square(gradients)
  gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
  gradient_l2_norm = K.sqrt(gradients_sqr_sum)
  gradient_penalty = K.square(1 - gradient_l2_norm)
  return K.mean(gradient_penalty)

# Weighted average function
class RandomWeightedAverage(Layer):
  def __init__(self, batch_size:int):
    super().__init__()
    self.batch_size = batch_size

  # Provides a (random) weighted average between real and generated image samples
  def call(self, inputs, **kwargs):
    weights = K.random_uniform((self.batch_size, 1, 1, 1))
    return (weights * inputs[0]) + ((1 - weights) * inputs[1])

  def compute_output_shape(self, input_shape):
    return input_shape[0]

class WGANGC:
  AGREGATE_STAT_INTERVAL = 1_000 # Interval of saving data
  RESET_SEEDS_INTERVAL = 5_000 # Interval of reseting seeds for random generators
  CHECKPOINT_SAVE_INTERVAL = 500 # Interval of saving checkpoint

  def __init__(self, dataset_path:str,
               gen_mod_name:str, critic_mod_name:str,
               latent_dim:int,
               training_progress_save_path:str,
               generator_optimizer:Optimizer=RMSprop(0.00005), critic_optimizer:Optimizer=RMSprop(0.00005),
               batch_size:int=32, buffered_batches:int=20,
               generator_weights:Union[str, None]=None, critic_weights:Union[str, None]=None,
               critic_gradient_penalty_weight:float=10,
               start_episode:int=0, load_from_checkpoint:bool=False,
               check_dataset:bool=True, num_of_loading_workers:int=8):

    self.critic_mod_name = critic_mod_name
    self.gen_mod_name = gen_mod_name
    self.latent_dim = latent_dim
    assert self.latent_dim > 0, Fore.RED + "Invalid latent dim" + Fore.RESET

    self.batch_size = batch_size
    assert self.batch_size > 0, Fore.RED + "Invalid batch size" + Fore.RESET

    self.progress_image_dim = (16, 9)

    if start_episode < 0: start_episode = 0
    self.episode_counter = start_episode

    # Initialize training data folder and logging
    self.training_progress_save_path = training_progress_save_path
    self.training_progress_save_path = os.path.join(self.training_progress_save_path, f"{self.gen_mod_name}__{self.critic_mod_name}__{self.latent_dim}")
    self.tensorboard = TensorBoardCustom(log_dir=os.path.join(self.training_progress_save_path, "logs"))

    # Create array of input image paths
    self.train_data = get_paths_of_files_from_path(dataset_path)
    assert self.train_data, Fore.RED + "Training dataset is not loaded" + Fore.RESET

    # Load one image to get shape of it
    tmp_image = cv.imread(self.train_data[0])
    self.image_shape = tmp_image.shape
    self.image_channels = self.image_shape[2]

    # Check image size validity
    if self.image_shape[0] < 4 or self.image_shape[1] < 4: raise Exception("Images too small, min size (4, 4)")

    # Check validity of datasets
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

    # Build critic block
    self.critic = self.build_critic(critic_mod_name)

    # Build generator block
    self.generator = self.build_generator(gen_mod_name)
    if self.generator.output_shape[1:] != self.image_shape: raise Exception("Invalid image input size for this generator model")

    #################################
    ### Create combined generator ###
    #################################
    # Create model inputs
    gen_latent_input = Input(shape=(self.latent_dim,), name="combined_generator_latent_input")

    # Create frozen version of critic
    self.critic.trainable = False
    for layer in self.critic.layers:
      layer.trainable = False

    # Generate images and evaluate them
    generated_images = self.generator(gen_latent_input)
    critic_gen_output = self.critic(generated_images)

    self.combined_generator_model = Model(inputs=[gen_latent_input],
                                          outputs=[critic_gen_output],
                                          name="combined_generator_model")
    self.combined_generator_model.compile(optimizer=generator_optimizer, loss=wasserstein_loss)

    ##############################
    ### Create combined critic ###
    ##############################
    # Create model inputs
    real_image_input = Input(shape=self.image_shape, name="combined_critic_real_image_input")
    critic_latent_input = Input(shape=(self.latent_dim,), name="combined_critic_latent_input")

    # Create frozen version of generator
    self.critic.trainable = True
    for layer in self.critic.layers:
      layer.trainable = True

    self.generator.trainable = False
    for layer in self.generator.layers:
      layer.trainable = False

    # Create fake image input (internal)
    generated_images_for_critic = self.generator(critic_latent_input)

    # Create critic output for each image "type"
    fake_out = self.critic(generated_images_for_critic)
    valid_out = self.critic(real_image_input)

    # Create weighted input to critic for gradient penalty loss
    averaged_samples = RandomWeightedAverage(self.batch_size)(inputs=[real_image_input, generated_images_for_critic])
    validity_interpolated = self.critic(averaged_samples)

    # Create partial gradient penalty loss function
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples)
    partial_gp_loss.__name__ = 'gradient_penalty'

    self.combined_critic_model = Model(inputs=[real_image_input, critic_latent_input],
                                       outputs=[valid_out,
                                                fake_out,
                                                validity_interpolated],
                                       name="combined_critic_model")
    self.combined_critic_model.compile(optimizer=critic_optimizer,
                                       loss=[wasserstein_loss,
                                             wasserstein_loss,
                                             partial_gp_loss],
                                       loss_weights=[1, 1, critic_gradient_penalty_weight])

    # Summary of combined models
    print("\nGenerator Summary:")
    self.generator.summary()
    print("\nCritic Summary:")
    self.critic.summary()

    # Load checkpoint
    self.initiated = False
    if load_from_checkpoint: self.load_checkpoint()

    # Load weights and override checkpoint loaded weights
    if critic_weights: self.critic.load_weights(critic_weights)
    if generator_weights: self.generator.load_weights(generator_weights)

    # Create batchmaker and start it
    self.batch_maker = BatchMaker(self.train_data, self.batch_size, buffered_batches=buffered_batches, num_of_loading_workers=num_of_loading_workers)

    # Create some proprietary objects
    self.fake_labels = np.ones((self.batch_size, 1), dtype=np.float32)
    self.valid_labels = -np.ones((self.batch_size, 1), dtype=np.float32)
    self.gradient_labels = np.zeros((self.batch_size, 1), dtype=np.float32)

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

    print(Fore.BLUE + "Dataset valid" + Fore.RESET)

  # Create generator based on template selected by name
  def build_generator(self, model_name:str):
    noise_input = Input(shape=(self.latent_dim,))

    try:
      m = getattr(generator_models_spreadsheet, model_name)(noise_input, self.image_shape, self.image_channels, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Generator model not found!\n{e}")
    return Model(noise_input, m, name="generator_model")

  # Create critic based on teplate selected by name
  def build_critic(self, model_name:str):
    img_input = Input(shape=self.image_shape)

    try:
      m = getattr(discriminator_models_spreadsheet, model_name)(img_input, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Critic model not found!\n{e}")

    # Linear output for critic
    m = Dense(1)(m)
    return Model(img_input, m, name="critic_model")

  def train(self, target_episode:int,
            progress_images_save_interval:int=None, save_raw_progress_images:bool=True, weights_save_interval:int=None,
            critic_train_multip:int=5):

    # Check arguments and input data
    assert target_episode > 0, Fore.RED + "Invalid number of episodes" + Fore.RESET
    if progress_images_save_interval:
      assert progress_images_save_interval <= target_episode, Fore.RED + "Invalid progress save interval" + Fore.RESET
    if weights_save_interval:
      assert weights_save_interval <= target_episode, Fore.RED + "Invalid weights save interval" + Fore.RESET
    assert critic_train_multip >= 1, Fore.RED + "Invalid critic training multiplier" + Fore.RESET

    # Calculate epochs to go
    end_episode = target_episode
    target_episode = target_episode - self.episode_counter
    assert target_episode > 0, Fore.CYAN + "Training is already finished" + Fore.RESET

    # Save noise for progress consistency
    if progress_images_save_interval is not None:
      if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)
      np.save(f"{self.training_progress_save_path}/static_noise.npy", self.static_noise)

    epochs_time_history = deque(maxlen=self.AGREGATE_STAT_INTERVAL * 50)

    # Save starting kernels and biases
    if not self.initiated:
      self.__save_imgs(save_raw_progress_images)
      self.save_checkpoint()

    print(Fore.GREEN + f"Starting training on episode {self.episode_counter} for {target_episode} episodes" + Fore.RESET)
    for _ in range(target_episode):
      ep_start = time.time()

      ### Train Critic ###
      critic_loss = 0
      for _ in range(critic_train_multip):
        # Load image batch and generate new latent noise
        image_batch = self.batch_maker.get_batch()
        critic_noise_batch = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

        critic_loss += float(self.combined_critic_model.train_on_batch([image_batch, critic_noise_batch], [self.valid_labels, self.fake_labels, self.gradient_labels])[0])
      critic_loss /= critic_train_multip

      ### Train Generator ###
      # Generate new latent noise
      gen_loss = self.combined_generator_model.train_on_batch(np.random.normal(0.0, 1.0, (self.batch_size, self.latent_dim)), self.valid_labels)

      self.episode_counter += 1
      self.tensorboard.step = self.episode_counter
      self.tensorboard.update_stats(critic_loss=critic_loss, gen_loss=gen_loss)

      # Show stats
      if self.episode_counter % self.AGREGATE_STAT_INTERVAL == 0:
        # Save stats
        print(Fore.GREEN + f"{self.episode_counter}/{end_episode}, Remaining: {time_to_format(mean(epochs_time_history) * (end_episode - self.episode_counter))}\t\t[Critic loss: {round(float(critic_loss), 5)}] [Gen loss: {round(float(gen_loss), 5)}]" + Fore.RESET)

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

      # Reset seeds
      if self.episode_counter % self.RESET_SEEDS_INTERVAL == 0:
        np.random.seed(None)
        random.seed()

      epochs_time_history.append(time.time() - ep_start)

    # Shutdown helper threads
    print(Fore.GREEN + "Training Complete - Waiting for other threads to finish" + Fore.RESET)
    self.batch_maker.terminate()
    self.save_checkpoint()
    self.__save_weights()
    self.batch_maker.join()
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

  def save_models_structure_images(self, save_path:str=None):
    if save_path is None: save_path = self.training_progress_save_path + "/model_structures"
    if not os.path.exists(save_path): os.makedirs(save_path)
    plot_model(self.combined_generator_model, os.path.join(save_path, "combined_generator.png"), expand_nested=True, show_shapes=True)
    plot_model(self.combined_critic_model, os.path.join(save_path, "combined_critic.png"), expand_nested=True, show_shapes=True)
    plot_model(self.generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)
    plot_model(self.critic, os.path.join(save_path, "critic.png"), expand_nested=True, show_shapes=True)

  def load_checkpoint(self):
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
          self.critic.load_weights(data["critic_path"])
        except:
          print(Fore.YELLOW + "Failed to load critic weights from checkpoint" + Fore.RESET)

        self.initiated = True

  def save_checkpoint(self):
    checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
    if not os.path.exists(checkpoint_base_path): os.makedirs(checkpoint_base_path)

    gen_path = f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5"
    critic_path = f"{checkpoint_base_path}/discriminator_{self.critic_mod_name}.h5"

    if os.path.exists(gen_path): os.rename(gen_path, f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5.lock")
    if os.path.exists(critic_path): os.rename(critic_path, f"{checkpoint_base_path}/discriminator_{self.critic_mod_name}.h5.lock")

    self.generator.save_weights(gen_path)
    self.critic.save_weights(critic_path)

    if os.path.exists(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5.lock")
    if os.path.exists(f"{checkpoint_base_path}/discriminator_{self.critic_mod_name}.h5.lock"): os.remove(f"{checkpoint_base_path}/discriminator_{self.critic_mod_name}.h5.lock")

    data = {
      "episode": self.episode_counter,
      "gen_path": gen_path,
      "critic_path": critic_path
    }

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "w", encoding='utf-8') as f:
      json.dump(data, f)

  def __save_weights(self):
    save_dir = self.training_progress_save_path + "/weights/" + str(self.episode_counter)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")
    self.critic.save_weights(f"{save_dir}/critic_{self.critic_mod_name}.h5")

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