import os
import sys
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdin = stdin
sys.stderr = stderr

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
			tf.config.experimental.set_memory_growth(gpus[0], True)
	except:
		pass

from keras import optimizers
from modules.dcgan import DCGAN
from modules.wasserstein_gan import WGANGC
from modules.srgan import SRGAN
from modules.ssrgan import SSRGAN
from settings import *

if __name__ == '__main__':
	gan = None
	try:
		gan_selection = int(input("GAN selection\n0 - DCGAN\n1 - WGAN\n2 - SRGAN\n3 - SSRGAN\nSelected GAN: "))
		if gan_selection == 0:
			gan = DCGAN(DATASET_PATH, training_progress_save_path="training_data/dcgan",
			            batch_size=BATCH_SIZE, buffered_batches=BUFFERED_BATCHES, test_batches=5,
			            latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, disc_mod_name=DISC_MODEL,
			            generator_optimizer=optimizers.Adam(0.0002, 0.5), discriminator_optimizer=optimizers.Adam(0.00018, 0.5),
			            discriminator_label_noise=0.2, discriminator_label_noise_decay=0.997, discriminator_label_noise_min=0.03,
			            generator_weights=GEN_WEIGHTS, discriminator_weights=DICS_WEIGHTS,
			            start_episode=START_EPISODE,
			            load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
			            pretrain=5,
			            check_dataset=CHECK_DATASET)

			gan.save_models_structure_images()

			while True:
				gan.train(NUM_OF_TRAINING_EPOCHS, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
				          weights_save_interval=WEIGHTS_SAVE_INTERVAL,
				          discriminator_smooth_real_labels=True, discriminator_smooth_fake_labels=False,
				          generator_smooth_labels=False,
				          feed_prev_gen_batch=True, feed_old_perc_amount=0.15)
				if input("Continue? ") == "n": break

		elif gan_selection == 1:
			gan = WGANGC(DATASET_PATH, training_progress_save_path="training_data/wgan",
			             batch_size=BATCH_SIZE, buffered_batches=BUFFERED_BATCHES,
			             latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, critic_mod_name=DISC_MODEL,
			             generator_optimizer=optimizers.RMSprop(0.00005), critic_optimizer=optimizers.RMSprop(0.00005),  # Adam(0.0001, beta_1=0.5, beta_2=0.9), RMSprop(0.00005)
			             generator_weights=GEN_WEIGHTS, critic_weights=DICS_WEIGHTS,
			             critic_gradient_penalty_weight=10,
			             start_episode=START_EPISODE,
			             load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
			             check_dataset=CHECK_DATASET)

			gan.save_models_structure_images()

			while True:
				gan.train(NUM_OF_TRAINING_EPOCHS, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
				          weights_save_interval=WEIGHTS_SAVE_INTERVAL,
				          critic_train_multip=5)
				if input("Continue? ") == "n": break

		elif gan_selection == 2:
			gan = SRGAN(DATASET_SR_PATH, num_of_upscales=NUM_OF_UPSCALES, training_progress_save_path="training_data/srgan",
			            batch_size=BATCH_SIZE_SR, buffered_batches=BUFFERED_BATCHES_SR,
			            gen_mod_name=GEN_SR_MODEL, disc_mod_name=DISC_SR_MODEL,
			            generator_optimizer=optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), discriminator_optimizer=optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
			            discriminator_label_noise=None, discriminator_label_noise_decay=0.995, discriminator_label_noise_min=0.03,
			            generator_weights=GEN_SR_WEIGHTS, discriminator_weights=DICS_SR_WEIGHTS,
			            start_episode=START_EPISODE_SR,
			            load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
			            custom_batches_per_epochs=CUSTOM_BATCHES_PER_EPOCH, check_dataset=CHECK_DATASET)

			gan.save_models_structure_images()

			while True:
				gan.train(NUM_OF_TRAINING_EPOCHS_SR, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
				          weights_save_interval=WEIGHTS_SAVE_INTERVAL,
				          discriminator_smooth_real_labels=True, discriminator_smooth_fake_labels=True,
				          generator_smooth_labels=True,
				          pretrain_epochs=5)
				if input("Continue? ") == "n": break

		elif gan_selection == 3:
			gan = SSRGAN(DATASET_SR_PATH, num_of_upscales=NUM_OF_UPSCALES, training_progress_save_path="training_data/ssrgan",
			             batch_size=BATCH_SIZE_SR, buffered_batches=BUFFERED_BATCHES_SR,
			             gen_mod_name=GEN_SR_MODEL,
			             generator_optimizer=optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
			             generator_weights=GEN_SR_WEIGHTS,
			             start_episode=START_EPISODE_SR,
			             load_from_checkpoint=LOAD_FROM_CHECKPOINTS,
			             custom_batches_per_epochs=CUSTOM_BATCHES_PER_EPOCH, check_dataset=CHECK_DATASET)

			gan.save_models_structure_images()

			while True:
				gan.train(NUM_OF_TRAINING_EPOCHS_SR, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_raw_progress_images=SAVE_RAW_IMAGES,
				          weights_save_interval=WEIGHTS_SAVE_INTERVAL)
				if input("Continue? ") == "n": break

		gan.save_weights()
		gan.save_checkpoint()
	except KeyboardInterrupt:
		if gan:
			print(f"Quiting on epoch: {gan.epoch_counter} - This could take little time, get some coffe and rest :)")
			gan.save_checkpoint()
	except Exception as e:
		if gan:
			print(f"Exception on epoch: {gan.epoch_counter}")
			gan.save_checkpoint()
		else:
			print(f"Creating GAN failed")
		traceback.print_exc()