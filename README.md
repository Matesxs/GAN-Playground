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
* [Results](#results)
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
Conditional GAN - GAN for generating new images from latent vector and labels
Pix2Pix using GAN - Model for transforming a image
CycleGAN - GAN for transforming a image
ProGAN - GAN for generating new images from latent vector, using progressive growing models and GP loss
```

## Project Folder Structure
```
- gans (scripts for each GAN)
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

## Utility
```
preprocess_dataset.py - Script for mass rescaling images to target size and optionaly splitting them to training and testing parts
clean_dataset.py - Clean low resolution images from dataset
```

## Results

### DCGAN
1. Mnist dataset (64x64 grayscale) - 20 epochs, batch size 128 \
   ![1](media/dcgan/mnist_dcgan_fake.png?raw=true)

2. Celeb dataset (64x64 color, 200000 images) - batch size 128 \
   Unstable training and colapsed after few more epochs \
   No need for more training, because its by design prone to fails   

   1. Generated - 100 epochs \
   ![2](media/dcgan/faces_dcgan_fake.png?raw=true)
   2. Colapsed network (epoch 110) \
   ![3](media/dcgan/faces_dcgan_fake_colapsed.png?raw=true)

### WGAN
1. Celeb dataset (64x64 color, 200000 images) - batch size 64
   1. Generated - 30 epochs - TODO: More training \
   ![4](media/dcgan/faces_wgan_fake1.png?raw=true)

### WGAN-GP
1. Celeb dataset (64x64 color, 200000 images) - batch size 64
   1. Generated - 30 epochs - TODO: More training \
   ![5](media/dcgan/faces_wgan-gp_fake1.png?raw=true)

### Conditional GAN - Based on WGAN-GP
1. Mnist dataset (64x64 grayscale) - batch size 64 - TODO: Revisit
   1. Original model
      1. Generated - 20 epochs \
      ![6](media/dcgan/mnist_cond-gan_fake.png?raw=true)
      2. Real \
      ![7](media/dcgan/mnist_cond-gan_real.png?raw=true)
   2. Larger model
      1. Generated - 20 epochs \
      ![8](media/dcgan/mnist_cond-gan_fake2.png?raw=true)
      2. Real \
      ![9](media/dcgan/mnist_cond-gan_real2.png?raw=true)

### Pix2Pix using GAN
1. Maps segmentation (256x256 color, 2000 images) - batch size 16, 200 epochs \
In order: Input, Real, Generated
![10](media/dcgan/maps_pix2pix_input.png?raw=true)
![11](media/dcgan/maps_pix2pix_real.png?raw=true)
![12](media/dcgan/maps_pix2pix_fake.png?raw=true)

2. Anime coloring (256x256 color, 16000 images) - batch size 16, 400 epochs \
In order: Input, Real, Generated
![13](media/dcgan/anime_pix2pix_input.png?raw=true)
![14](media/dcgan/anime_pix2pix_real.png?raw=true)
![15](media/dcgan/anime_pix2pix_fake.png?raw=true)

## TODO
- [x] Implement basic DCGAN
- [x] Implement basic WGAN
- [x] Implement WGAN-GP (WGAN with gradient penalty)
- [x] Implement basic Conditional GAN
- [x] Implement Pix2Pix based on GAN
- [x] Implement basic CycleGAN
- [x] Implement basic ProGAN
- [x] Implement basic SRGAN
- [ ] Implement enhanced SRGAN (ESRGAN)
- [ ] Implement basic StyleGAN

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
https://www.alexirpan.com/2017/02/22/wasserstein-gan.html \
https://arxiv.org/pdf/1701.07875.pdf \
https://arxiv.org/pdf/1704.00028.pdf \
https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
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
##### Pix2Pix using GAN
https://arxiv.org/pdf/1611.07004.pdf
<br/>
<br/>
##### CycleGAN
https://arxiv.org/pdf/1703.10593.pdf
<br/>
<br/>
##### ProGAN
https://arxiv.org/pdf/1710.10196.pdf
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

### Some resources might be missing, I started researching this topic long before this repository was created!
