import os
import sys

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

'''
Generators:
	mod_base_3upscl - New high capacity
	mod_base_3upscl_test - added leaky
	mod_ext_3upscl
	
Discriminators:
	mod_ext_5layers
	mod_ext_5layers_test
	mod_base_8layers - Experimental model from stylegan
	
Settings testing:
	|       Gen       |       Disc        | Lat. Dim | Epochs | Rank | Description
	mod_base_3upscl     mod_ext_5layers     128
	mod_ext_3upscl      mod_base_8layers    128       
'''

DATASET_PATH = "dataset/normalized_dogs"
LATENT_DIM = 128

GEN_MODEL = "mod_ext_3upscl"
DISC_MODEL = "mod_base_8layers"

NUM_OF_EPISODES = 100

WEIGHTS_SAVE_INTERVAL = 5
PROGRESS_IMAGE_SAVE_INTERVAL = 1
SAVE_TRAINING_STATS = True

if __name__ == '__main__':
	gan = None
	try:
		gan_selection = int(input("GAN selection\n0 - DCGAN\n1 - WGAN\nSelected GAN: "))
		if gan_selection == 0:
			gan = DCGAN(DATASET_PATH, training_progress_save_path="training_data/dcgan", progress_image_dim=(16, 9),
			            batch_size=8,
			            latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, disc_mod_name=DISC_MODEL,
			            generator_optimizer=optimizers.Adam(0.0002, 0.5), discriminator_optimizer=optimizers.Adam(0.0002, 0.5),
			            discriminator_label_noise=0.2, discriminator_label_noise_decay=0.9998, discriminator_label_noise_min=0.01,
			            generator_weights=None, discriminator_weights=None,
			            start_episode=0,
			            load_from_checkpoint=True,
			            pretrain=5)

			gan.save_models_structure_images()

			while True:
				gan.train(NUM_OF_EPISODES, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_training_stats=SAVE_TRAINING_STATS, buffered_batches=100,
				          weights_save_interval=WEIGHTS_SAVE_INTERVAL,
				          discriminator_smooth_real_labels=True, discriminator_smooth_fake_labels=False,
				          feed_prev_gen_batch=True, feed_amount=0.1)
				if input("Continue? ") == "n": break
		elif gan_selection == 1:
			gan = WGANGC(DATASET_PATH, training_progress_save_path="training_data/wgan", progress_image_dim=(16, 9),
			             batch_size=8,
			             latent_dim=LATENT_DIM, gen_mod_name=GEN_MODEL, critic_mod_name=DISC_MODEL,
			             generator_optimizer=optimizers.RMSprop(0.00005), critic_optimizer=optimizers.RMSprop(0.00005),  # Adam(0.0001, beta_1=0.5, beta_2=0.9), RMSprop(0.00005)
			             generator_weights=None, critic_weights=None,
			             critic_gradient_penalty_weight=10,
			             start_episode=0,
			             load_from_checkpoint=False)

			gan.save_models_structure_images()

			while True:
				gan.train(NUM_OF_EPISODES, progress_images_save_interval=PROGRESS_IMAGE_SAVE_INTERVAL, save_training_stats=SAVE_TRAINING_STATS, buffered_batches=100,
			            weights_save_interval=WEIGHTS_SAVE_INTERVAL,
			            critic_train_multip=5)
				if input("Continue? ") == "n": break

		gan.save_weights()
		gan.save_checkpoint()
	except KeyboardInterrupt:
		if gan:
			print(f"Quiting on epoch: {gan.epoch_counter} - This could take little time, get some coffe and rest :)")
			gan.save_checkpoint()
	except Exception as e:
		if gan:
			print(f"Exception on epoch: {gan.epoch_counter}\n{e}")
			gan.save_checkpoint()
		else:
			print(f"Creating GAN failed\n{e}")

	if gan:
		# gan.show_training_stats()
		gan.show_training_stats(save=True)

		if input("Make progress gif?\n") == "y": gan.make_progress_gif(framerate=10)
		if input("Generate collage?\n") == "y": gan.generate_collage(collage_dims=(16, 9))