---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: MobileNet v2
summary: The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models.
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: mobilenet_v2_1.png
featured_image_2: mobilenet_v2_2.png
---

### Model Description

The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input. MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, non-linearities in the narrow layers were removed in order to maintain representational power.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  mobilenet_v2       | 28.12       | 9.71       |


### Notes on Inputs

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

### Example:

```python
import torch
model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
```

### Resources:

 - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
