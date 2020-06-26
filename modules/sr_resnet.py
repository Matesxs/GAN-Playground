from colorama import Fore
import os
from typing import Union
import keras.backend as K
from keras.optimizers import Optimizer, Adam
from keras.layers import Input
from keras.models import Model
from keras.applications.vgg19 import VGG19
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

from modules.models import upscaling_generator_models_spreadsheet
from modules.custom_tensorboard import TensorBoardCustom
from modules.batch_maker import BatchMaker
from modules.helpers import time_to_format

# Calculate start image size based on final image size and number of upscales
def count_upscaling_start_size(target_image_shape: tuple, num_of_upscales: int):
  upsc = (target_image_shape[0] // (2 ** num_of_upscales), target_image_shape[1] // (2 ** num_of_upscales), target_image_shape[2])
  if upsc[0] < 1 or upsc[1] < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
  return upsc

class VGG_LOSS(object):
  _IMAGENET_MEAN = K.constant(-np.array([103.939, 116.778, 123.68]))

  def __init__(self, image_shape):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
      l.trainable = False
    self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block2_conv2").output)
    self.model.trainable = False

  def preproces_vgg(self, x):
    # scale from [-1,1] to [0, 255]
    x += 1.
    x *= 127.5

    # RGB -> BGR
    x = x[..., ::-1]

    # apply Imagenet preprocessing : BGR mean
    x = K.bias_add(x, K.cast(self._IMAGENET_MEAN, K.dtype(x)))

    return x

  # computes VGG loss or content loss
  def vgg_loss(self, y_true, y_pred):
    features_pred = self.model(self.preproces_vgg(y_pred))
    features_true = self.model(self.preproces_vgg(y_true))

    return 0.006 * K.mean(K.square(features_pred - features_true), axis=-1)

class SR_Resnet:
  AGREGATE_STAT_INTERVAL = 1  # Interval of saving data
  RESET_SEEDS_INTERVAL = 10  # Interval of checking norm gradient value of combined model
  CHECKPOINT_SAVE_INTERVAL = 1  # Interval of saving checkpoint

  def __init__(self, dataset_path: str, num_of_upscales: int,
               gen_mod_name: str,
               training_progress_save_path: str,
               generator_optimizer: Optimizer = Adam(0.0002, 0.5),
               batch_size: int = 32, buffered_batches: int = 20,
               generator_weights: Union[str, None, int] = None,
               start_episode: int = 0, load_from_checkpoint: bool = False,
               custom_batches_per_epochs: int = None, custom_hr_test_image_path:str=None, check_dataset: bool = True):

    self.gen_mod_name = gen_mod_name
    self.num_of_upscales = num_of_upscales
    assert self.num_of_upscales >= 0, Fore.RED + "Invalid number of upscales" + Fore.RESET

    self.batch_size = batch_size
    assert self.batch_size > 0, Fore.RED + "Invalid batch size" + Fore.RESET

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
    self.training_progress_save_path = os.path.join(self.training_progress_save_path, f"{self.gen_mod_name}__{self.start_image_shape}_to_{self.target_image_shape}")
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

    # Loss object
    self.loss_object = VGG_LOSS(self.target_image_shape)

    #################################
    ###     Create generator      ###
    #################################
    self.generator = self.build_generator(gen_mod_name)
    if self.generator.output_shape[1:] != self.target_image_shape: raise Exception("Invalid image input size for this generator model")
    self.generator.compile(loss=self.loss_object.vgg_loss, optimizer=generator_optimizer)
    print("\nGenerator Sumary:")
    self.generator.summary()

    # Load checkpoint
    self.initiated = False
    if load_from_checkpoint: self.load_checkpoint()

    # Load weights from param and override checkpoint weights
    if generator_weights: self.generator.load_weights(generator_weights)

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
  def build_generator(self, model_name: str):
    small_image_input = Input(shape=self.start_image_shape)

    try:
      m = getattr(upscaling_generator_models_spreadsheet, model_name)(small_image_input, self.start_image_shape, self.num_of_upscales, self.kernel_initializer)
    except Exception as e:
      raise Exception(f"Generator model not found!\n{e}")

    return Model(small_image_input, m, name="generator_model")

  def train(self, epochs: int, progress_images_save_interval: int = None, save_raw_progress_images: bool = True, weights_save_interval: int = None):

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
    if not self.initiated:
      self.__save_img(save_raw_progress_images)
      self.tensorboard.log_kernels_and_biases(self.generator)
      self.save_checkpoint()

    print(Fore.GREEN + f"Starting training on epoch {self.epoch_counter} for {epochs} epochs" + Fore.RESET)
    for _ in range(epochs):
      ep_start = time.time()
      for _ in tqdm(range(num_of_batches), unit="batches", smoothing=0.5, leave=False):
        ### Train Generator ###
        # Train generator (wants discriminator to recognize fake images as valid)
        large_images, small_images = self.batch_maker.get_batch()
        self.generator.train_on_batch(small_images, large_images)

      time.sleep(0.5)
      self.epoch_counter += 1
      epochs_time_history.append(time.time() - ep_start)
      self.tensorboard.step = self.epoch_counter

      # Seve stats and print them to console
      if self.epoch_counter % self.AGREGATE_STAT_INTERVAL == 0:
        large_images, small_images = self.batch_maker.get_batch()

        # Evaluate model state
        gen_loss = self.generator.test_on_batch(small_images, large_images)

        self.tensorboard.log_kernels_and_biases(self.generator)
        self.tensorboard.update_stats(self.epoch_counter, gen_loss=gen_loss)

        print(Fore.GREEN + f"{self.epoch_counter}/{end_epoch}, Remaining: {time_to_format(mean(epochs_time_history) * (end_epoch - self.epoch_counter))} - [G loss: {round(float(gen_loss), 5)}]" + Fore.RESET)

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

  def __save_img(self, save_raw_progress_images: bool = True):
    if not os.path.exists(self.training_progress_save_path + "/progress_images"): os.makedirs(self.training_progress_save_path + "/progress_images")
    if not os.path.exists(self.progress_test_image_path):
      print(Fore.YELLOW + "Test image doesnt exist anymore, choosing new one" + Fore.RESET)
      self.progress_test_image_path = random.choice(self.train_data)
      self.save_checkpoint()

    original_image = cv.imread(self.progress_test_image_path)
    gen_img = self.generator.predict(np.array([cv.cvtColor(cv.resize(original_image, dsize=(self.start_image_shape[0], self.start_image_shape[1]), interpolation=(cv.INTER_AREA if (original_image.shape[0] > self.start_image_shape[0] and original_image.shape[1] > self.start_image_shape[1]) else cv.INTER_CUBIC)), cv.COLOR_BGR2RGB) / 127.5 - 1.0]))[0]

    # Rescale images 0 to 255
    gen_img = (0.5 * gen_img + 0.5) * 255
    gen_img = cv.cvtColor(gen_img, cv.COLOR_RGB2BGR)

    final_image = np.zeros(shape=(gen_img.shape[0], gen_img.shape[1] * 2, gen_img.shape[2])).astype(np.float32)
    final_image[:, 0:gen_img.shape[0], :] = original_image
    final_image[:, gen_img.shape[0]:gen_img.shape[0] * 2, :] = gen_img

    if save_raw_progress_images:
      cv.imwrite(f"{self.training_progress_save_path}/progress_images/{self.epoch_counter}.png", final_image)
    self.tensorboard.write_image(np.reshape(cv.cvtColor(final_image, cv.COLOR_BGR2RGB) / 255, (-1, final_image.shape[0], final_image.shape[1], final_image.shape[2])).astype(np.float32))

  def save_weights(self):
    save_dir = self.training_progress_save_path + "/weights/" + str(self.epoch_counter)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    self.generator.save_weights(f"{save_dir}/generator_{self.gen_mod_name}.h5")

  def save_models_structure_images(self):
    save_path = self.training_progress_save_path + "/model_structures"
    if not os.path.exists(save_path): os.makedirs(save_path)
    plot_model(self.generator, os.path.join(save_path, "generator.png"), expand_nested=True, show_shapes=True)

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

        if not self.custom_hr_test_image_path:
          self.progress_test_image_path = data["test_image"]
        self.initiated = True

  def save_checkpoint(self):
    checkpoint_base_path = os.path.join(self.training_progress_save_path, "checkpoint")
    if not os.path.exists(checkpoint_base_path): os.makedirs(checkpoint_base_path)

    self.generator.save_weights(f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5")

    data = {
      "episode": self.epoch_counter,
      "gen_path": f"{checkpoint_base_path}/generator_{self.gen_mod_name}.h5",
      "test_image": self.progress_test_image_path
    }

    with open(os.path.join(checkpoint_base_path, "checkpoint_data.json"), "w", encoding='utf-8') as f:
      json.dump(data, f)

  def make_progress_gif(self, frame_duration: int = 16):
    if not os.path.exists(self.training_progress_save_path + "/progress_images"): return
    if not os.path.exists(self.training_progress_save_path): os.makedirs(self.training_progress_save_path)

    frames = []
    img_file_names = os.listdir(self.training_progress_save_path + "/progress_images")

    for im_file in img_file_names:
      if os.path.isfile(self.training_progress_save_path + "/progress_images/" + im_file):
        frames.append(Image.open(self.training_progress_save_path + "/progress_images/" + im_file))

    if len(frames) > 2:
      frames[0].save(f"{self.training_progress_save_path}/progress_gif.gif", format="GIF", append_images=frames[1:], save_all=True, optimize=False, duration=frame_duration, loop=0)