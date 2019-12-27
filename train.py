from src.dcgan import DCGAN

'''
Generators:
	mod_base_2upscl - Bad on color images, some tweaks needed, "good" on small gray images
	mod_base_3upscl - Kind of work, some "good" results after 1000 epochs - 1024 latent dim
	mod_base_4upscl - Bad decaying loss at 300 epoch, maybe more training needed or some tweaks
	mod_ext_4upscl - Works fine (200 epochs, 512 latent dim)
	
Discriminators:
	mod_base_4layers - Works fine
	mod_base_5layers - Maybe works but need more testing
	mod_min_4layers - Not tested
'''

if __name__ == '__main__':
	gan = DCGAN("training_data/normalized", progress_image_path="prog_images",
	            latent_dim=512, gen_mod_name="mod_ext_4upscl", disc_mod_name="mod_min_4layers")
	gan.save_models_structure_images()
	# gan.show_sample_of_dataset()
	gan.clear_progress_images()

	# Training with showing progress
	while True:
		gan.train(200, 32, progress_save_interval=10,
		          weights_save_path="trained_weights", weights_save_interval=10)
		gan.show_current_state(3)
		gan.show_training_stats()

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_gif()
	if input("Generate images?\n") == "y": gan.generate_random_images(100, "gen_images")