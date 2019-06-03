---
layout: hub_detail
background-class: hub-background
body-class: hub
title: FCN-ResNet101
summary: Fully-Convolutional Network model with a ResNet-101 backbone
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [vision]
github-link: https://github.com/pytorch/vision.git
featured_image_1: no-image
featured_image_2: no-image
---

### Model Description

FCN-ResNet101 is contructed by a Fully-Covolutional Network model with a ResNet-101 backbone.
The pre-trained models have been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.

Their accuracies of the pre-trained models evaluated on COCO val2017 dataset are listed below.

| Model structure |   Mean IOU  | Global Pixelwise Accuracy |
| --------------- | ----------- | --------------------------|
|  fcn_resnet101  |   63.7      |   91.9                    |

### Notes on Inputs

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

### Example

```python
import torch
model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
```

### Resources

 - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211)
