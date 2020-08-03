import os
import sys
import traceback
import colorama
import subprocess
from colorama import Fore

colorama.init()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdin = stdin
sys.stderr = stderr

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except:
    pass

from keras import optimizers
from modules.gans.dcgan import DCGAN
from modules.gans.wasserstein_gan import WGANGC
from modules.gans.srgan import SRGAN
from settings import *

if __name__ == '__main__':
  training_object = None
  if not os.path.exists("training_data"): os.makedirs("training_data")
  tbmanager = subprocess.Popen("./venv/Scripts/python.exe -m tensorboard.main --logdir training_data --samples_per_plugin=images=200", stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  try:
    gan_selection = int(input("Trainer selection\n0 - DCGAN\n1 - WGAN\n2 - SRGAN\nSelected trainer: "))
    if gan_selection == 0:
      training_object = DCGAN(DATASET_PATH, testing_dataset_path=TESTING_DATASET_PATH, training_progress_save_path="training_data/dcgan",
                              batch_size=BATCH_SIZE, buffered_batches=BUFFERED_BATCHES,
                              latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, disc_mod_name=DISC_MODEL,
                              generator_optimizer=optimizers.Adam(0.0002, 0.5), discriminator_optimizer=optimizers.Adam(0.00018, 0.5),
                              discriminator_label_noise=0.2, discriminator_label_noise_decay=0.997, discriminator_label_noise_min=0.03,
                              generator_weights=GEN_WEIGHTS, discriminator_weights=DICS_WEIGHTS,
                              start_episode=START_EPISODE,
                              load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
                              pretrain_episodes=500_000,
                              check_dataset=CHECK_DATASET)

      training_object.save_models_structure_images()

      while True:
        training_object.train(NUM_OF_TRAINING_EPISODES, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
                              weights_save_interval=WEIGHTS_SAVE_INTERVAL,
                              discriminator_smooth_real_labels=True, discriminator_smooth_fake_labels=False,
                              generator_smooth_labels=False,
                              feed_prev_gen_batch=True, feed_old_perc_amount=0.15)
        if input("Continue? ") == "n": break

    elif gan_selection == 1:
      training_object = WGANGC(DATASET_PATH, testing_dataset_path=TESTING_DATASET_PATH, training_progress_save_path="training_data/wgan",
                               batch_size=BATCH_SIZE, buffered_batches=BUFFERED_BATCHES,
                               latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, critic_mod_name=DISC_MODEL,
                               generator_optimizer=optimizers.RMSprop(0.00005), critic_optimizer=optimizers.RMSprop(0.00005),  # Adam(0.0001, beta_1=0.5, beta_2=0.9), RMSprop(0.00005)
                               generator_weights=GEN_WEIGHTS, critic_weights=DICS_WEIGHTS,
                               critic_gradient_penalty_weight=10,
                               start_episode=START_EPISODE,
                               load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
                               check_dataset=CHECK_DATASET)

      training_object.save_models_structure_images()

      while True:
        training_object.train(NUM_OF_TRAINING_EPISODES, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
                              weights_save_interval=WEIGHTS_SAVE_INTERVAL,
                              critic_train_multip=5)
        if input("Continue? ") == "n": break

    elif gan_selection == 2:
      training_object = SRGAN(DATASET_SR_PATH, testing_dataset_path=TESTING_DATASET_SR_PATH, num_of_upscales=NUM_OF_UPSCALES, training_progress_save_path="training_data/srgan",
                              batch_size=BATCH_SIZE_SR, testing_batch_size=TESTING_BATCH_SIZE_SR, buffered_batches=BUFFERED_BATCHES_SR,
                              gen_mod_name=GEN_SR_MODEL, disc_mod_name=DISC_SR_MODEL,
                              generator_optimizer=optimizers.Adam(GEN_LR_SRGAN, 0.9), discriminator_optimizer=optimizers.Adam(DISC_LR_SRGAN, 0.9),
                              generator_lr_schedule=GEN_LR_SCHEDULE_SRGAN, discriminator_lr_schedule=DISC_LR_SCHEDULE_SRGAN,
                              discriminator_label_noise=DISCRIMINATOR_START_NOISE_OF_SRGAN, discriminator_label_noise_decay=DISCRIMINATOR_NOISE_DECAY_OF_SRGAN, discriminator_label_noise_min=DISCRIMINATOR_TARGET_NOISE_OF_SRGAN,
                              generator_weights=GEN_SR_WEIGHTS, discriminator_weights=DICS_SR_WEIGHTS,
                              start_episode=START_EPISODE_SR,
                              load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
                              custom_hr_test_image_path=CUSTOM_HR_TEST_IMAGE, check_dataset=CHECK_DATASET)

      training_object.save_models_structure_images()

      training_object.train(COMBINED_TRAINING_EPISODES_SRGAN, generator_train_episodes=GENERATOR_TRAIN_EPISODES_OF_SRGAN, discriminator_train_episodes=DISCRIMINATOR_TRAIN_EPISODES_OF_SRGAN,
                            discriminator_training_multiplier=DISCRIMINATOR_TRAINING_MULTIPLIER,
                            progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL_SR, save_raw_progress_images=SAVE_RAW_IMAGES,
                            weights_save_interval=WEIGHTS_SAVE_INTERVAL,
                            discriminator_smooth_real_labels=True, discriminator_smooth_fake_labels=True,
                            generator_smooth_labels=False,
                            training_autobalancer=AUTOBALANCE_TRAINING_OF_SRGAN, save_only_best_pnsr_weights=SAVE_ONLY_BEST_PNSR_WEIGHTS)

    else: print(Fore.RED + "Invalid training object index entered" + Fore.RESET)
  except KeyboardInterrupt:
    if training_object:
      print(Fore.BLUE + f"Quiting on epoch: {training_object.episode_counter} - This could take little time, get some coffe and rest :)" + Fore.RESET)
  except Exception as e:
    if training_object:
      print(Fore.RED + f"Exception on epoch: {training_object.episode_counter}" + Fore.RESET)
    else:
      print(Fore.RED + "Creating training object failed" + Fore.RESET)
    traceback.print_exc()
  finally:
    if training_object:
      training_object.save_checkpoint()

  if training_object:
    if input("Create gif of progress? ") == "y": training_object.make_progress_gif(frame_duration=GIF_FRAME_DURATION)

  try:
    tbmanager.send_signal(subprocess.signal.CTRL_C_EVENT)
    tbmanager.wait()
  except:
    pass