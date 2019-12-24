from src.dcgan import DCGAN

if __name__ == '__main__':
	gan = DCGAN("training_data/normalized", latent_dim=512, progress_image_path="prog_images", gen_v=1, disc_v=1)
	gan.plot_models()
	# gan.show_sample_of_dataset()
	gan.clear_output_folder()

	# Training with showing progress
	while True:
		gan.train(50, 32, 5, smooth=0.1, trick_fake=True)
		gan.show_current_state(5)
		gan.show_training_stats()
		gan.save_weights("trained_weights")

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_gif()
	if input("Generate images?\n") == "y": gan.generate_random_images(100, "gen_images")