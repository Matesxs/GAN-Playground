from models.dcgan import DCGAN

if __name__ == '__main__':
	gan = DCGAN("training_data/normalized", latent_dim=512, gen_v=1, disc_v=1)
	gan.plot_models()
	# gan.show_sample_of_dataset()
	gan.clear_output_folder()

	# Training with showing progress
	for _ in range(4):
		gan.train(50, 32, 10, smooth=0.1, trick_fake=True)
		gan.show_current_state(5)
		gan.show_training_stats()
		gan.save_models_prompt()

	# gan.make_gif()