---
layout: pytorch_hub_detail
background-class: hub-background
body-class: hub
title: GoogLeNet
summary: GoogLeNet was based on a deep convolutional neural network architecture codenamed "Inception" which won ImageNet 2014.
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: googlenet1.png
featured_image_2: googlenet2.png
---

### Model Description

GoogLeNet was based on a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The 1-crop error rates on the ImageNet dataset with a pretrained model are list below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  googlenet       | 30.22       | 10.47       |


### Notes on Inputs

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

### Example

```python
import torch
model = torch.hub.load('pytorch/vision', 'googlenet', pretrained=True)
```

### Resources

 - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
