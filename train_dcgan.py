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

from modules.gans import DCGAN
from modules.utils.helpers import start_tensorboard
from settings.dcgan_settings import *

if __name__ == '__main__':
  training_object = None
  if not os.path.exists("training_data/dcgan"): os.makedirs("training_data/dcgan")
  bmanager = start_tensorboard("training_data/dcgan")

  try:
    training_object = DCGAN(DATASET_PATH, training_progress_save_path="training_data/dcgan",
                            batch_size=BATCH_SIZE, buffered_batches=BUFFERED_BATCHES,
                            latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, disc_mod_name=DISC_MODEL,
                            generator_optimizer=optimizers.Adam(0.0001, 0.5), discriminator_optimizer=optimizers.Adam(0.0001, 0.5),
                            discriminator_label_noise=0.2, discriminator_label_noise_decay=0.997, discriminator_label_noise_min=0.03,
                            generator_weights=GEN_WEIGHTS, discriminator_weights=DICS_WEIGHTS,
                            start_episode=START_EPISODE,
                            load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
                            check_dataset=CHECK_DATASET, num_of_loading_workers=NUM_OF_LOADING_WORKERS)

    training_object.save_models_structure_images()

    training_object.train(NUM_OF_TRAINING_EPISODES, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
                          weights_save_interval=WEIGHTS_SAVE_INTERVAL,
                          discriminator_smooth_real_labels=True, discriminator_smooth_fake_labels=False,
                          generator_smooth_labels=False,
                          feed_prev_gen_batch=True, feed_old_perc_amount=0.15)
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