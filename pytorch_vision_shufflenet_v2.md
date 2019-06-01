---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ShuffleNet v2
summary: Based on a series of controlled experiments, this work derives several practical guidelines for efficient network design.
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: shufflenet_v2_1.png
featured_image_2: shufflenet_v2_2.png
---

### Model Description

Previously, neural network architecture design was mostly guided by the indirect metric of computation complexity, i.e., FLOPs. However, the direct metric, e.g., speed, also depends on the other factors such as memory access cost and platform characteristics. Based on a series of controlled experiments, this work derives several practical guidelines for efficient network design. Accordingly, a new architecture is presented, called ShuffleNet V2. Comprehensive ablation experiments verify that our model is the state of-the-art in terms of speed and accuracy tradeoff.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  shufflenet_v2       | 30.64       | 11.68       |


### Notes on Inputs

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

### Example

```python
import torch
model = torch.hub.load('pytorch/vision', 'shufflenet_v2_x1_0', pretrained=True)
```

### Resources

 - [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
