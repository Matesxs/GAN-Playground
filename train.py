from keras.optimizers import Adam
from modules.dcgan import DCGAN

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
	mod_base_2upscl     mod_base_4layers    100        100000   D      Not enough capacity
	mod_min_3upscl      mod_min_5layers     100
	mod_base_3upscl     mod_ext_5layers     100   --- Maybe the best combination, but models are too large ---
'''

if __name__ == '__main__':
	gan = DCGAN("dataset/cats/normalized", training_progress_save_path="training_data", progress_image_num=10,
	            latent_dim=100, gen_mod_name="mod_min_3upscl", disc_mod_name="mod_min_5layers",
	            generator_optimizer=Adam(0.0002, 0.5), discriminator_optimizer=Adam(0.0002, 0.5),
	            generator_weights=None, discriminator_weights=None)
	gan.clear_training_progress_folder()
	gan.save_models_structure_images()
	# gan.show_sample_of_dataset(10)

	# Training with showing progress
	while True:
		try:
			gan.train(100_000, 32, progress_images_save_interval=200, agregate_stats_interval=100,
		            weights_save_interval=None,
		            discriminator_smooth_labels=True, generator_smooth_labels=True, discriminator_label_noise=0.02,
		            feed_prev_gen_batch=False, feed_amount=0.1)
		except KeyboardInterrupt:
			gan.show_training_stats(save_path="training_data")

		gan.show_current_state(3, 5)
		gan.show_training_stats(save_path=None)
		gan.show_training_stats(save_path="training_data")

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_progress_gif(save_path="training_data")
	if input("Generate collage?\n") == "y": gan.generate_collage(save_path="training_data", collage_dims=(16, 9))