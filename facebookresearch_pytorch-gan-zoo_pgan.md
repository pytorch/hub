---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: Progressive Growing of GAN (PGAN)
summary: An implementation of NVIDIA's method for generating HD images with GAN https://arxiv.org/abs/1710.10196
category: researchers
image: pytorch-logo.png
author: FAIR HDGAN
tags: [GAN, Vision, HD, Celeba, CelebaHQ]
github-link: https://github.com/facebookresearch/pytorch_GAN_zoo
featured_image_1: pgan_mix.jpg
featured_image_2: pgan_celebaHQ.jpg
---

<!-- REQUIRED: detailed model description below, in markdown format, feel free to add new sections as necessary -->
### Model Description

In computer vision, generative models are networks trained to create images from a given input. In our case, we consider a specific kind of generative networks: GANs (Generative Adversarial Networks) which learn to map a random vector with a realistic image generation.

Progressive Growing of GAN is a method developed by NVIDIA in 2017 allowing to generate high resolution images: https://arxiv.org/abs/1710.10196. To do so, the generative network is trained slice by slice. At first the model is trained to build very low resolution images, once it converges, new layers are added and the output resolution doubles. The process continues until the desired resolution is reached.

<!-- REQUIRED: provide a working script to demonstrate it works with torch.hub -->
### Example

In order to load and use a model trained on CelebaHQ:

```python
import torch
  import scipy.misc
  import numpy as np

  def saveTensor(data, path):
      data = (torch.clamp(data, min=-1, max=1) + 1.0) * 255.0 / 2.0
      scipy.misc.imsave(path, np.array(data.permute(1,2,0).numpy(),
                                       dtype='uint8'))

  model = torch.hub.load('facebookresearch/pytorch_GAN_zoo',
                         'PGAN',
                         pretrained=True, model_name="celebAHQ-512")

  batch_size = 4
  inputRandom, _ = model.buildNoiseData(batch_size)

  outImgs = model.test(inputRandom, getAvG=True, toCPU=True)
  for index in range(batch_size):
      saveTensor(outImgs[index], f"test_{index}.jpg")
```

Other model names are available: celebAHQ-256, DTD (https://www.robots.ox.ac.uk/~vgg/data/dtd/) and celeba (res 128).
If you want to train your own model please have a look at https://github.com/facebookresearch/pytorch_GAN_zoo.


<!-- OPTIONAL: put special requirement of your model here, e.g. only supports Python3 -->
### Requirement

The model only support python3.
