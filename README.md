# GAN Testing Playground (WIP)
#### Project about testing techniques about training GANs and their stability

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Usage](#usage)
* [Utility](#utility)
* [Results](#results)
* [Used models](#used-models)
* [TODO](#todo)
* [Current tasks](#current-tasks)
* [Notes](#notes)
* [Testing setup](#testing-setup)
* [Resources](#resources)

## General info
This project contains documented code for DCGAN, WGAN and SRGAN. \
GAN and WGAN are for creating new unique images from latent vector. \
Sometimes somebody could refed to it as noise but in general its more like settings values. \
SRGAN is more useful GAN, its purpose is to upscale image from low to higher resolution.

## Setup
```
pip install -r requirements.txt
```

## Usage
```
Adjust settings in settings/****_settings.py
Download some datasets and place them in dataset directory (Or in directory you set in settings.py)
python preprocess_dataset.py
python train_****.py

After training use
1) python generate_images.py for DCGAN and WGAN
2) python upscale_images.py for SRGAN
(These scripts still needs tweaking because settings for them are hardcoded in them)

Note: These scripts are still in progress of making, some of them may not work!
```

## Utility
```
scrape_lorem_picsum.py - Script for scraping lorem picsum like websites
preprocess_dataset.py - Script for mass rescaling images to target size and optionaly splitting them to training and testing parts
```

## Results
##### SRGAN Results - (Upscaled by opencv, Original, Upscaled by SRGAN)
1) Model without using bias (batch normalization "bias" is used instead) and without using spectral normalization \
1 400 000 episodes
![SRGAN_image_1](./images/srganResultImage1.png?raw=true)
2) Model using bias and spectral normalization \
2 500 000 episodes
![SRGAN_image_2](./images/srganResultImage2.png?raw=true)

## Used models
```
    Generator / Discriminator (Critic)
1. mod_srgan_exp / mod_base_9layers
2. mod_srgan_exp_sn / mod_base_9layers_sn
```

## TODO
- [x] Implement DCGAN
- [x] Implement WGAN
- [x] Implement SRGAN
- [ ] Implement StyleGAN
- [x] Implement custom Tensorboard for logging stats
- [x] Implement learning rate scheduler
- [x] Implement spectral normalization layer
- [x] Test Charbonnier loss instead of MSE loss in SRGAN
- [ ] Test Huber loss instead of MSE loss in SRGAN
- [x] Test MAE loss instead of MSE loss in SRGAN
- [ ] Optimize training loop of SRGAN (Too much time is spending of testing performance)
- [x] Implement custom batch maker
- [ ] Optimize batch maker to use generator class from keras
- [x] Optimize preprocessing dataset (Too slow)
- [ ] Optimize interface scripts with more acessible settings
- [ ] Retrain all SRGAN models with single test image with same train settings to properly compare them

## Current tasks
```
- Testing best working model pairs for WGAN
- Optimizations of SRGAN
- Better interface scripts
- Refactoring
```

## Notes
Testing of Charbonnier loss for SRGAN failed because the values were too different from MSE loss values, maybe more tweaking required and test again. \
MAE loss is causing lot of artifacts and image distortions (like color shifting, "image bleedoff", etc) in results from SRGAN.

## Testing setup
```
Hardware:
    Processor: I7-9700KF 4.8GHz
    RAM: HyperX Fury RGB 32GB (2x16GB) DDR4 3200MHz
    GPU: GIGABYTE GeForce RTX 2080 SUPER 8G
    SSD: Intel 660p M.2 2TB SSD NVMe

Editor: PyCharm (always latest version)

Important libraries:
    tensorflow==2.2.0
    keras==2.3.1
    numpy==1.19.1

These versions of libraries were tested for these scripts.
```

## Resources
##### Basic DCGAN
https://github.com/mitchelljy/DCGAN-Keras \
https://arxiv.org/pdf/1511.06434.pdf
<br/>
<br/>
##### WGAN (Wasserstein GAN)
https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/ \
https://arxiv.org/pdf/1704.00028.pdf
<br/>
<br/>
##### WGAN-GP
https://github.com/LuEE-C/WGAN-GP-with-keras-for-text/blob/master/Exploration/GenerativeAdverserialWGAN-GP.py \
https://github.com/kongyanye/cwgan-gp \
https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py \
https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py \
https://github.com/keras-team/keras/issues/8585 \
Note: Some of these concepts were used in my implementation of my WGAN
<br/>
<br/>
##### SRGAN (Super Resolution GAN)
https://arxiv.org/pdf/1609.04802.pdf \
https://github.com/deepak112/Keras-SRGAN/blob/master/Network.py \
https://medium.com/@birla.deepak26/single-image-super-resolution-using-gans-keras-aca310f33112 \
https://github.com/MathiasGruber/SRGAN-Keras
<br/>
<br/>
##### SR Resnet
https://github.com/JGuillaumin/SuperResGAN-keras \
Note: Implemented as pretrain for SRGAN
<br/>
<br/>
##### Perceptual Loss
https://arxiv.org/pdf/1603.08155.pdf%7C \
https://deepai.org/machine-learning-glossary-and-terms/perceptual-loss-function \
https://github.com/yuqil/style_transfer_keras/blob/master/vgg/vgg_loss.py
<br/>
<br/>
##### GAN Stability and diagnostics
https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/ \
https://github.com/soumith/ganhacks \
https://medium.com/intel-student-ambassadors/tips-on-training-your-gans-faster-and-achieve-better-results-9200354acaa5 \
https://distill.pub/2016/deconv-checkerboard/