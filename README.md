# GAN Testing Playground (WIP)
#### Project about testing techniques about training GANs and their stability

<img align="left" src=https://img.shields.io/github/license/Matesxs/GAN-Playground>
<img align="left" src=https://img.shields.io/github/stars/Matesxs/GAN-Playground>
<img align="left" src=https://img.shields.io/github/forks/Matesxs/GAN-Playground>
<img align="left" src=https://img.shields.io/github/issues/Matesxs/GAN-Playground>
<br/>

## Table of contents
* [General info](#general-info)
* [Content](#content)
* [Project Folder Structure](#project-folder-structure)
* [Setup](#setup)
* [Usage](#usage)
* [Results](#results)
* [TODO](#todo)
* [Current tasks](#current-tasks)
* [Notes](#notes)
* [Testing setup](#testing-setup)
* [Resources](#resources)

## General info
This project contains code for some of the most know types of GAN (Generative Adverserial Network).
I am using this repo to play with these types of networks to get better understanding how they work and how to properly train them.

Disclaimer: This repository is more like proof of concept than download and run!

## Content
```
DCGAN - GAN for generating new images from latent vector
WGAN(GC) - GAN for generating new images from latent vector
SRGAN - GAN for upscaling images
```

## Project Folder Structure
```
- datasets (place for all data that you will want feed to network)
- media (folder with media files of repo)
- modules
    - gans (trainers for all GANs are pleced)
    - keras_extensions (extesions based on keras functionality)
    - models (models and building blocks for models)
    - utils (other helper stuff)
- settings (settings for scripts)
- utility_scripts (scripts for data processing and other useful stuff)
```

## Setup
```
pip install -r requirements.txt
```

## Dependencies
```
- Python3.7
- Tensorflow 2.2.0
- Keras 2.3.1

For GPU acceleration:
    - Cuda Toolkit 10.1
    - cuDNN for toolkit version
```

## Usage
```
Adjust settings in settings/****_settings.py
Get datasets and place them in dataset directory (Or in directory you set in settings.py)
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
preprocess_dataset.py - Script for mass rescaling images to target size and optionaly splitting them to training and testing parts
visualize_conv_activations.py - Script for displaying activation of each conv layer as image
show_vgg_structure.py - Script that will print all layers of vgg19 usable for perceptual loss
parse_hr_image.py - Script to parse large images to small ones (WIP)
Note: Some utility scripts have its settings in settings folder
```

## Results
##### SRGAN Results - (Upscaled by opencv, Original, Upscaled by SRGAN)
###### Pretrain
For my dataset ideal pretrain of generator is something around 50k episodes \
1) No pretrain, 400k episodes \
![SRGAN_image_1](media/srgan_results/srgan_no_pretrain.png?raw=true)
2) 50k pretrain, 400k episodes \
![SRGAN_image_2](media/srgan_results/sragan_50k_pretrain.png?raw=true)
3) 200k pretrain, 400k episodes \
![SRGAN_image_2](media/srgan_results/sragan_200k_pretrain.png?raw=true)

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
- [ ] Test impact of different weights of losses in SRGAN
- [x] Optimize training loop of SRGAN (Too much time is spending of testing performance)
- [x] Implement custom batch maker
- [ ] Optimize batch maker to use generator class from keras
- [x] Optimize preprocessing dataset (Too slow)
- [x] Optimize interface scripts with more acessible settings
- [x] Test pretrain effect on results from SRGAN
- [ ] Retrain all SRGAN models with single test image with same train settings to properly compare them
- [ ] Implement gradient accumulation to "simulate" larger batch

## Current tasks
```
- Testing best working model pairs for WGAN
- Refactoring
- Retraining all SRGAN models
- Testing efects of pretrain on SRGAN model
```

## Notes
Testing of Charbonnier loss for SRGAN failed because the values were too different from MSE loss values, maybe more tweaking required and test again. \
MAE loss is causing lot of artifacts and image distortions (like color shifting, "image bleedoff", etc) in results from SRGAN. \

## Testing setup
```
Hardware:
    Processor: I7-9700KF 4.8GHz
    RAM: HyperX Fury RGB 32GB (2x16GB) DDR4 3200MHz
    GPU: GIGABYTE GeForce RTX 2080 SUPER 8G
    SSD: Intel 660p M.2 2TB SSD NVMe

Editor: PyCharm (always latest version)
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
https://github.com/MathiasGruber/SRGAN-Keras \
https://github.com/idealo/image-super-resolution
<br/>
<br/>
##### SR Resnet
https://github.com/JGuillaumin/SuperResGAN-keras
<br/>
<br/>
##### ESDR (Enhanced Deep Residual Networks for Single Image Super-Resolution)
https://github.com/Golbstein/EDSR-Keras \
Note: Ideas used for improving SRGAN
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
<br/>
<br/>
##### Gradient accumulation
https://stackoverflow.com/questions/55268762/how-to-accumulate-gradients-for-large-batch-sizes-in-keras \
https://github.com/keras-team/keras/issues/3556#issuecomment-440638517

### Some resources might be missing, I started researching this topic long before this repository was created!
