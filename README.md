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
Disclaimer: This repository is more like proof of concept than download and run! \
Some scripts might not work because I refactoring so fast and forgot to test it.

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
- gans (scripts for each GAN except training scripts)
- datasets (place for all data that you will want feed to network)
- media (folder with media files of repo)
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
clean_small_images.py - Clean low resolution images from dataset
```

## Results

### DCGAN
1. Mnist dataset (64x64 grayscale) - 100k iters, batch size 128 \
   ![1](media/mnist_dcgan_fake.png?raw=true)

2. Celeb dataset (64x64 color, 200000 images) - batch size 128 \
   Unstable training and colapsed after few more epochs \
   No need for more training, because its by design prone to fails   

   1. Generated - 3M iters \
   ![2](media/faces_dcgan_fake.png?raw=true)
   2. Colapsed network (3.3M iters) \
   ![3](media/faces_dcgan_fake_colapsed.png?raw=true)

### WGAN

More stable training in comparison to DCGAN but slower to train and capacity of model is smaller because of hard clamping weights

1. Celeb dataset (64x64 color, 200000 images) - batch size 64
   1. Generated - 3M iters \
   ![4](media/faces_wgan_fake1.png?raw=true)
   2. Generated - 6M iters \
   ![5](media/faces_wgan_fake2.png?raw=true)
2. Celeb dataset (64x64 color, 200000 images) - batch size 64 \
   Model with replaced batch norm lazers with instance norm layers \
   Stability of model is improved
   1. Generated - 600k iters \
   ![6](media/faces_wgan_fake3.png?raw=true)
   2. Generated - 6M iters \
   ![8](media/faces_wgan_fake4.png?raw=true)

### WGAN-GP
1. Celeb dataset (64x64 color, 200000 images) - batch size 64 \
   Generated - 500k iters \
   ![9](media/faces_wgan-gp_fake1.png?raw=true)
2. SOCOFing dataset (64x64 gray, 6000 images) - batch size 32 \
   1. Generated 100k iters \
   ![10](media/socofing_wgan-gp_fake1.png?raw=true)
   2. Generated 100k iters - pixel suffle \
   ![11](media/socofing_wgan-gp_fake2.png?raw=true)

### Conditional GAN - Based on WGAN-GP
1. Mnist dataset (64x64 grayscale) - batch size 64
   1. Generated - 200k iters \
   ![12](media/mnist_cond-gan_fake.png?raw=true)
   2. Real \
   ![13](media/mnist_cond-gan_real.png?raw=true)

### Pix2Pix using GAN
1. Maps segmentation (256x256 color, 2000 images) - batch size 16, 200 epochs \
In order: Input, Real, Generated
![14](media/maps_pix2pix_input.png?raw=true)
![15](media/maps_pix2pix_real.png?raw=true)
![16](media/maps_pix2pix_fake.png?raw=true)

2. Anime coloring (256x256 color, 16000 images) - batch size 16, 300k iters \
In order: Input, Real, Generated
![17](media/anime_pix2pix_input.png?raw=true)
![18](media/anime_pix2pix_real.png?raw=true)
![19](media/anime_pix2pix_fake.png?raw=true)

3. Fingerprint correction (256x256 color, 6000 images) - batch size 8, 400k iters \
In order: Input, Real, Generated
![20](media/fingerprint_pix2pix_input.png?raw=true)
![21](media/fingerprint_pix2pix_original.png?raw=true)
![22](media/fingerprint_pix2pix_generated.png?raw=true)

### ProGAN
1. Generated 16x16 \
![23](media/celeba_progan_fake1.png?raw=true)
2. Generated 32x32 \
![24](media/celeba_progan_fake2.png?raw=true)

## TODO
- [x] Implement basic DCGAN
- [x] Implement basic WGAN
- [x] Implement WGAN-GP (WGAN with gradient penalty)
- [x] Implement basic Conditional GAN
- [x] Implement Pix2Pix based on GAN
- [x] Implement basic CycleGAN
- [ ] Revisit CycleGAN with different dataset
- [x] Implement basic ProGAN
- [x] Fix ProGAN and revisit for better results
- [ ] Try ProGAN with classic WGAN model layers
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
https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
<br/>
<br/>
##### WGAN-GP
https://arxiv.org/pdf/1704.00028.pdf \
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
https://arxiv.org/pdf/1710.10196.pdf \
https://github.com/nvnbny/progressive_growing_of_gans
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
##### Inception score
https://github.com/sbarratt/inception-score-pytorch \
https://arxiv.org/pdf/1801.01973.pdf

### Some resources might be missing, I started researching this topic long before this repository was created!
