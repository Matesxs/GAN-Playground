# GAN Testing Playground
#### Project about testing techniques about training GANs and their stability

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Usage](#usage)
* [Utility Scripts](#utility)
* [TODO](#todo)

## General info
This project contains documented code for DCGAN, WGAN and SRGAN.
GAN and WGAN are for creating new unique images from latent vector.
Sometimes somebody could refed to it as noise but in general its more like settings values.
SRGAN is more useful GAN, its purpose is to upscale image from low to higher resolution.

## Setup
```
pip install -r requirements.txt
```

## Usage
```
Adjust settings in settings.py
Download some datasets and place them in dataset directory (Or in directory you set in settings.py)
python preprocess_dataset.py
python train.py

After training use
1) python generate_images.py for DCGAN and WGAN
2) python upscale_images.py for SRGAN
(These scripts still needs tweaking because settings for them are hardcoded in them)
```

## Utility
```
scrape_lorem_picsum.py - Script for scraping lorem picsum like websites
preprocess_dataset.py - Script for mass rescaling images to target size and optionaly splitting them to training and testing parts
```

## TODO
- [x] Implement DCGAN
- [x] Implement WGAN
- [x] Implement SRGAN
- [ ] Test stability of DCGAN
- [ ] Test stability of WGAN
- [x] Test stability of SRGAN
- [x] Implement custom Tensorboard for logging stats
- [x] Implement learning rate scheduler
- [x] Implement spectral normalization layer
- [ ] Test Charbonnier loss instead of MSE loss in SRGAN
- [ ] Test MAE loss instead of MSE loss in SRGAN
- [ ] Optimize training loop of SRGAN (Too much time is spending of testing performance)
- [x] Implement custom batch maker
- [ ] Optimize batch maker to use generator class from keras
- [ ] Optimize preprocessing dataset (Too slow)
- [ ] Test difference between models with conv2 layers with enabled and disabled bias
- [ ] Add sources to this readme
- [ ] Optimize interface scripts with more acessible settings