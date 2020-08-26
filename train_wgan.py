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
from modules.gans.wasserstein_gan import WGANGC
from settings.wgan_settings import *

if __name__ == '__main__':
  training_object = None
  if not os.path.exists("training_data"): os.makedirs("training_data")
  tbmanager = subprocess.Popen("./venv/Scripts/python.exe -m tensorboard.main --logdir training_data --samples_per_plugin=images=200", stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  try:
    training_object = WGANGC(DATASET_PATH, training_progress_save_path="training_data/wgan",
                             batch_size=BATCH_SIZE, buffered_batches=BUFFERED_BATCHES,
                             latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, critic_mod_name=DISC_MODEL,
                             generator_optimizer=optimizers.RMSprop(0.00005), critic_optimizer=optimizers.RMSprop(0.00005),  # Adam(0.0001, beta_1=0.5, beta_2=0.9), RMSprop(0.00005)
                             generator_weights=GEN_WEIGHTS, critic_weights=DICS_WEIGHTS,
                             critic_gradient_penalty_weight=10,
                             start_episode=START_EPISODE,
                             load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
                             check_dataset=CHECK_DATASET, num_of_loading_workers=NUM_OF_LOADING_WORKERS)

    training_object.save_models_structure_images()

    training_object.train(NUM_OF_TRAINING_EPISODES, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
                          weights_save_interval=WEIGHTS_SAVE_INTERVAL,
                          critic_train_multip=5)
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