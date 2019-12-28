from keras.optimizers import Adam
from modules.dcgan import DCGAN
from modules import generator_models_spreadsheet, discriminator_models_spreadsheet

save_path = "models_testing"
training_epochs = 200

latent_dims = [ 128, 256, 512, 1024 ]
batch_sizes = [ 16, 32, 64, 128 ]

generator_models = [name for name in dir(generator_models_spreadsheet) if name.startswith("mod_")]
discriminator_models = [name for name in dir(discriminator_models_spreadsheet) if name.startswith("mod_")]

for gen_model in generator_models:
	for disc_model in discriminator_models:
		for latent_dim in latent_dims:
			for batch_size in batch_sizes:
				testing_name = f"{gen_model}-{disc_model}-{latent_dim}ld-{batch_size}bs"
				gan = DCGAN("training_data/normalized", progress_image_path=f"{save_path}/{testing_name}/progress_images", progress_image_num=10,
				            latent_dim=latent_dim, gen_mod_name=gen_model, disc_mod_name=disc_model,
				            generator_optimizer=Adam(0.0002, 0.5), discriminator_optimizer=Adam(0.0002, 0.5),
				            generator_weights=None, discriminator_weights=None)
				gan.clear_progress_images()
				gan.train(training_epochs, batch_size, progress_save_interval=10,
				          discriminator_smooth_labels=True, generator_smooth_labels=True,
				          feed_prew_gen_batch=True)
				gan.show_training_stats(plt_save_path=f"{save_path}/{testing_name}")
				gan.generate_collage(save_path=f"{save_path}/{testing_name}")