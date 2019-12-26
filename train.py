from src.dcgan import DCGAN

if __name__ == '__main__':
	gan = DCGAN("training_data/normalized", latent_dim=512, progress_image_path="prog_images", gen_mod_name="mod_ext_4upscl", disc_mod_name="mod_min_4layers")
	gan.plot_models()
	# gan.show_sample_of_dataset()
	gan.clear_progress_images()

	# Training with showing progress
	for _ in range(10):
		gan.train(200, 32, 10, smooth=0.1, trick_fake=True, weights_save_path="trained_weights", weights_save_interval=20)
		gan.show_current_state(3)
		gan.show_training_stats()

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_gif()
	if input("Generate images?\n") == "y": gan.generate_random_images(100, "gen_images")