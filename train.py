from keras.optimizers import Adam
from modules.dcgan import DCGAN

'''
Generators:
	mod_base_2upscl
	mod_extM_2upscl
	mod_base_3upscl
	mod_base_4upscl
	mod_ext_4upscl
	
Discriminators:
	mod_base_4layers
	mod_extD_4layers
	mod_base_5layers
	mod_ext_5layers
	mod_ext2_5layers
	mod_extD_5layers
	mod_base_6layers
	mod_extD_6layers
	
Settings testing:
	|       Gen       |       Disc        | Lat. Dim | Epochs | Disc : Gen | Rank | Description
	mod_base_2upscl     mod_ext2_5layers    256        None     1:1          ---    Impossible to train now - too long (485s/ep)
	mod_base_2upscl     mod_extD_5layers    256        200      1:1          D      Artifacts - Maybe fixed, need test again
	mod_base_2upscl     mod_base_5layers    256        
'''

if __name__ == '__main__':
	gan = DCGAN("training_data/normalized", progress_image_path="prog_images", progress_image_num=10,
	            latent_dim=256, gen_mod_name="mod_base_2upscl", disc_mod_name="mod_base_5layers",
	            generator_optimizer=Adam(0.0002, 0.5), discriminator_optimizer=Adam(0.0002, 0.5),
	            generator_weights=None, discriminator_weights=None)
	gan.save_models_structure_images()
	# gan.show_sample_of_dataset(10)
	gan.clear_progress_images()

	# Training with showing progress
	while True:
		gan.train(100, 32, progress_save_interval=10,
		          weights_save_path="trained_weights", weights_save_interval=None,
		          discriminator_smooth_labels=True, generator_smooth_labels=True,
		          feed_prev_gen_batch=True, disc_train_multip=1, disc_half_batch=False)
		gan.show_current_state(3, 5)
		gan.show_training_stats(save_path=None)

		if input("Continue?\n") == "n": break

	if input("Make progress gif?\n") == "y": gan.make_gif()
	if input("Generate collage?\n") == "y": gan.generate_collage()