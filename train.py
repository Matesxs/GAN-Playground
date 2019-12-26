from src.dcgan import DCGAN

if __name__ == '__main__':
	gan = DCGAN("training_data/normalized", latent_dim=1024, progress_image_path="prog_images", gen_v=4, disc_v=2)
	gan.plot_models()
	# gan.show_sample_of_dataset()
	gan.clear_progress_images()

	# Training with showing progress
	while True:
		gan.train(100, 32, 5, smooth=0.1, trick_fake=True, weights_save_path="trained_weights", weights_save_interval=10)
		gan.show_current_state(5)
		gan.show_training_stats()
		gan.save_weights("trained_weights")

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_gif()
	if input("Generate images?\n") == "y": gan.generate_random_images(100, "gen_images")