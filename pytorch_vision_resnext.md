---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ResNext
summary: Simple, highly modularized network archi-tecture for image classification
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: resnext.png
featured_image_2: no-image
---

### Model Description

Resnext models were proposed in [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
Here we have the 2 versions of resnet models, which contains 50, 101 layers repspectively.
A comparison in model archetechure between resnet50 and resnext50 can be found in Table 1.
Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

|  Model structure  | Top-1 error | Top-5 error |
| ----------------- | ----------- | ----------- |
|  resnext50_32x4d  | 22.38       | 6.30        |
|  resnext101_32x8d | 20.69       | 5.47        |

### Notes on Inputs

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

### Example

```python
import torch
model = torch.hub.load('pytorch/vision', 'resnext50_32x4d', pretrained=True)
model = torch.hub.load('pytorch/vision', 'resnext101_32x8d', pretrained=True)
```

### Resources

 - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
