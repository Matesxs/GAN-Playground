from keras.optimizers import Adam
from modules.dcgan import DCGAN

'''
Generators:
	mod_base_2upscl
	mod_mod_2upscl
	mod_base_3upscl - New high capacity
	mod_min_3upscl  - Min version
	
Discriminators:
	mod_base_4layers
	mod_base_5layers
	mod_ext_5layers
	
Settings testing:
	|       Gen       |       Disc        | Lat. Dim | Epochs | Rank | Description
	mod_base_2upscl     mod_base_4layers    100       100000    D      Not enough capacity
	mod_base_3upscl     mod_ext_5layers     100
'''

if __name__ == '__main__':
	gan = DCGAN("dataset/normalized", training_progress_save_path="training_data", progress_image_num=10,
	            latent_dim=100, gen_mod_name="mod_min_3upscl", disc_mod_name="mod_base_5layers",
	            generator_optimizer=Adam(0.0002, 0.5), discriminator_optimizer=Adam(0.0002, 0.5),
	            generator_weights=None, discriminator_weights=None)
	gan.clear_training_progress_folder()
	gan.save_models_structure_images()
	# gan.show_sample_of_dataset(10)

	# Training with showing progress
	while True:
		gan.train(100000, 32, progress_images_save_interval=200, agregate_stats_interval=100,
		          weights_save_interval=None,
		          discriminator_smooth_labels=True, generator_smooth_labels=True, discriminator_label_noise=0.01,
		          feed_prev_gen_batch=False)
		gan.show_current_state(3, 5)
		gan.show_training_stats(save_path=None)
		gan.show_training_stats(save_path="training_data")

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_progress_gif(save_path="training_data")
	if input("Generate collage?\n") == "y": gan.generate_collage(save_path="training_data")