from keras import optimizers
from modules.dcgan import DCGAN
from modules.wasserstein_gan import WGANGC

'''
Generators:
	mod_base_2upscl
	mod_base_3upscl - New high capacity
	mod_min_3upscl  - Min version
	
Discriminators:
	mod_base_4layers
	mod_base_5layers
	mod_ext_5layers
	mod_min_5layers - Min version of ext
	mod_base_8layers - Experimental model from stylegan
	
Settings testing:
	|       Gen       |       Disc        | Lat. Dim | Epochs | Rank | Description
	mod_base_2upscl     mod_base_4layers    100        100000   D      Not enough capacity of models
	mod_min_3upscl      mod_min_5layers     100        500000   B
	mod_min_3upscl      mod_min_5layers     128        500000   B+
	mod_base_3upscl     mod_ext_5layers     100   --- Maybe the best combination, but models are too large for me ---
'''

if __name__ == '__main__':
	# Training with showing progress
	# This is loop training, you can do it at ones but meh, I dont like it
	gan = None
	try:
		gan_selection = int(input("GAN selection\n0 - DCGAN\n1 - WGAN\nSelected GAN: "))
		if gan_selection == 0:
			gan = DCGAN("dataset/normalized_dogs", training_progress_save_path="training_data/dcgan", progress_image_dim=(16, 9),
			            latent_dim=128, gen_mod_name="mod_base_3upscl", disc_mod_name="mod_ext_5layers",
			            generator_optimizer=optimizers.Adam(0.0002, 0.5), discriminator_optimizer=optimizers.Adam(0.0002, 0.5),
			            generator_weights=None, discriminator_weights=None,
			            start_episode=0)

			if input("Clear progress folder?\n") == "y": gan.clear_training_progress_folder()
			gan.save_models_structure_images()
			# gan.show_sample_of_dataset(10)

			gan.train(50_000, batch_size=16, progress_images_save_interval=200, save_training_stats=True, buffered_batches=40,
			          weights_save_interval=200,
			          discriminator_smooth_labels=True, generator_smooth_labels=True, discriminator_label_noise=0.06,
			          feed_prev_gen_batch=True, feed_amount=0.15)
		elif gan_selection == 1:
			gan = WGANGC("dataset/normalized_dogs", training_progress_save_path="training_data/wgan", progress_image_dim=(16, 9),
			             batch_size=16,
			             latent_dim=128, gen_mod_name="mod_base_3upscl", critic_mod_name="mod_ext_5layers",
			             generator_optimizer=optimizers.RMSprop(0.00005), critic_optimizer=optimizers.RMSprop(0.00005),  # Adam(0.0001, beta_1=0.5, beta_2=0.9), RMSprop(0.00005)
			             generator_weights=None, critic_weights=None,
			             critic_gradient_penalty_weight=10,
			             start_episode=0)

			if input("Clear progress folder?\n") == "y": gan.clear_training_progress_folder()
			gan.save_models_structure_images()
			# gan.show_sample_of_dataset(10)

			gan.train(50_000, progress_images_save_interval=200, save_training_stats=True, buffered_batches=40,
			          weights_save_interval=None,
			          critic_train_multip=5)

		gan.save_weights()
	except KeyboardInterrupt:
		if gan:
			print(f"Quiting on epoch: {gan.epoch_counter} - This could take little time, get some coffe and rest :)")
			gan.save_weights()
	except Exception as e:
		if gan:
			print(f"Exception on epoch: {gan.epoch_counter}\n{e}")
			if input("Save weights?\n") == "y": gan.save_weights()
		else:
			print(f"Creating GAN failed\n{e}")

	if gan:
		gan.show_current_state(3, 5)
		gan.show_training_stats()
		gan.show_training_stats(save=True)

		if input("Make progress gif?\n") == "y": gan.make_progress_gif(framerate=10)
		if input("Generate collage?\n") == "y": gan.generate_collage(collage_dims=(16, 9))