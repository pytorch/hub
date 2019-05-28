---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: resnet18
summary: Resnet18 is a 18-layers convolutional neural network that is trained from ImageNet database. The network can classify images into 1000 categories.
github-stars: 3436
category:
image: pytorch-logo.png
tags: [CV, image classification, python2, python3]
github-link: https://github.com/pytorch/vision.git
featured_image_1: resnet.png
featured_image_2:
---

Resnet18 is the simplest model proposed in "Deep Residual Learning for Image Recognition". It contains 18 layers as shown in Table 1.

### Examples:

```
import torch
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
```

### Resources:

 - "Deep Residual Learning for Image Recognition" on arXiv: <https://arxiv.org/abs/1512.03385>.
