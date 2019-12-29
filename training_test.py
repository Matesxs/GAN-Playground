from keras.optimizers import Adam
from modules.dcgan import DCGAN
from modules import generator_models_spreadsheet, discriminator_models_spreadsheet
import colorama
from colorama import Fore
import gc

save_path = "models_testing"
training_epochs = 200

latent_dims = [ 64, 128, 256 ]
batch_sizes = [ 16, 32, 64 ]

generator_models = [name for name in dir(generator_models_spreadsheet) if name.startswith("mod_")]
discriminator_models = [name for name in dir(discriminator_models_spreadsheet) if name.startswith("mod_")]

all_combinations = len(latent_dims) * len(batch_sizes) * len(generator_models) * len(discriminator_models)
done_tests = 0

colorama.init()
print(Fore.YELLOW + f"Number of tests: {all_combinations}")
print(Fore.RESET)
for gen_model in generator_models:
	for disc_model in discriminator_models:
		for latent_dim in latent_dims:
			for batch_size in batch_sizes:
				testing_name = f"{gen_model}-{disc_model}-{latent_dim}ld-{batch_size}bs"
				gan = DCGAN("dataset/normalized", training_progress_save_path=f"{save_path}/{testing_name}",
				            progress_image_num=10,
				            latent_dim=latent_dim, gen_mod_name=gen_model, disc_mod_name=disc_model,
				            generator_optimizer=Adam(0.0002, 0.5), discriminator_optimizer=Adam(0.0002, 0.5),
				            generator_weights=None, discriminator_weights=None)
				gan.clear_training_progress_folder()
				print(Fore.GREEN + f"{testing_name} - Test Started")
				print(Fore.RESET)
				gan.train(training_epochs, batch_size, progress_save_interval=10,
				          weights_save_interval=None,
					        discriminator_smooth_labels=True, generator_smooth_labels=True,
					        feed_prev_gen_batch=True)
				gan.show_training_stats(save_path=f"{save_path}/{testing_name}")
				gan.save_models_structure_images()
				gan.generate_collage(save_path=f"{save_path}/{testing_name}")
				print(Fore.GREEN + f"{testing_name} - Test Finished")

				del gan
				done_tests += 1
				print(Fore.YELLOW + f"Finished tests: {done_tests}/{all_combinations}")
				print(Fore.RESET)
				gc.collect()