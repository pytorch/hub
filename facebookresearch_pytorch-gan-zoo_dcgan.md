---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
category: researchers
title: Progressive Growing of GAN (PGAN)
summary: An implementation of DCGAN, a simple GAN model
image: pytorch-logo.png
author: FAIR HDGAN
tags: [GAN, vision, DCGAN]
github-link: https://github.com/facebookresearch/pytorch_GAN_zoo
featured_image_1: no-image
featured_image_2: no-image
---

### Model Description

In computer vision, generative models are networks trained to create images from a given input. In our case, we consider a specific kind of generative networks: GANs (Generative Adversarial Networks) which learn to map a random vector with a realistic image generation.

DCGAN is a model designed in 2015 by Radford et. al. in the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). It is a GAN architecture both very simple and efficient for low resolution image generation (up to 64x64).

### Example

In order to load and use a model trained on fashionGen:

```python
import torch
  import scipy.misc
  import numpy as np

  def saveTensor(data, path):
      data = (torch.clamp(data, min=-1, max=1) + 1.0) * 255.0 / 2.0
      scipy.misc.imsave(path, np.array(data.permute(1,2,0).numpy(),
                                       dtype='uint8'))

  model = torch.hub.load('facebookresearch/pytorch_GAN_zoo',
                         'DCGAN',
                         pretrained=True)

  batch_size = 4
  inputRandom, _ = model.buildNoiseData(batch_size)

  outImgs = model.test(inputRandom, getAvG=True, toCPU=True)
  for index in range(batch_size):
      saveTensor(outImgs[index], f"test_{index}.jpg")
```

If you want to train your own model please have a look at https://github.com/facebookresearch/pytorch_GAN_zoo.


### Requirement

The model only support python3.
