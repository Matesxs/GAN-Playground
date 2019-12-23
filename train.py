from models.dcgan import DCGAN

if __name__ == '__main__':
	# (x_train, _), (x_test, _) = cifar10.load_data()
	# x_train = np.concatenate((x_train, x_test))

	gan = DCGAN("data/normalized", latent_dim=512, gen_v=3, disc_v=1)
	gan.plot_models()
	# gan.show_sample_of_dataset()
	gan.clear_output_folder()
	gan.train(40, 64, 10, smooth=0.1, trick_fake=True)
	# gan.make_gif()
	gan.show_current_state(5)
	gan.show_training_stats()
	gan.save_models_prompt()