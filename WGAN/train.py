from keras.optimizers import RMSprop
from modules.wasserstein_gan import WGAN

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
	mod_min_3upscl      mod_min_5layers     128        
	mod_base_3upscl     mod_ext_5layers     128   --- Maybe the best combination, but models are too large for me ---
'''

if __name__ == '__main__':
	gan = WGAN("../dataset/cats/normalized", training_progress_save_path="training_data", progress_image_dim=(16, 9),
	            latent_dim=128, gen_mod_name="mod_min_3upscl", critic_mod_name="mod_min_5layers",
	            generator_optimizer=RMSprop(0.00005), critic_optimizer=RMSprop(0.00005),
	            generator_weights=None, critic_weights=None,
	            start_episode=0)
	if input("Clear progress folder?\n") == "y": gan.clear_training_progress_folder()
	gan.save_models_structure_images()
	# gan.show_sample_of_dataset(10)

	# Training with showing progress
	# This is loop training, you can do it at ones but meh, I dont like it
	while True:
		try:
			gan.train(20_000, 32, progress_images_save_interval=200, save_training_stats=True,
			          weights_save_interval=None,
			          critic_smooth_labels=True, generator_smooth_labels=True, critic_label_noise=0.05,
			          feed_prev_gen_batch=True, feed_amount=0.15, critic_multip=5)
			gan.save_weights()
		except KeyboardInterrupt:
			print(f"Quiting on epoch: {gan.epoch_counter} - This could take little time, get some coffe and rest :)")
			gan.save_weights()
			gan.show_training_stats(save_path="training_data")
		except Exception as e:
			print(f"Exception on epoch: {gan.epoch_counter}\n{e}")
			if input("Save weights?\n") == "y": gan.save_weights()

		gan.show_current_state(3, 5)
		gan.show_training_stats(save_path=None)
		gan.show_training_stats(save_path="training_data")

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_progress_gif(save_path="training_data", framerate=60)
	if input("Generate collage?\n") == "y": gan.generate_collage(save_path="training_data", collage_dims=(16, 9))