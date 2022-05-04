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

Previous version of this reposritory was moved to branch: old \
Disclaimer: This repository is more like proof of concept than download and run!

## Content
```
DCGAN - GAN for generating new images from latent vector
WGAN(GC) - GAN for generating new images from latent vector
```

## Project Folder Structure
```
- datasets (place for all data that you will want feed to network)
- media (folder with media files of repo)
- utility_scripts (scripts for data processing and other useful stuff)
```

## Setup
```
pip install -r requirements.txt
```

## Dependencies
```
- Python3.10
- PyTorch 1.11.0
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
```

## Results

## TODO
- [ ] Implement basic DCGAN
- [ ] Implement basic WGAN
- [ ] Implement WGAN-GP (WGAN with gradient penalty)

## Current tasks
```
- Testing best working model pairs for WGAN
- Refactoring
- Retraining all SRGAN models
- Testing efects of pretrain on SRGAN model
```

## Notes
Testing of Charbonnier loss for SRGAN failed because the values were too different from MSE loss values, maybe more tweaking required and test again. \
MAE loss is causing a lot of artifacts and image distortions (like color shifting, "image bleedoff", etc.) in results from SRGAN. \

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
https://github.com/Golbstein/EDSR-Keras
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
