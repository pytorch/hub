---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: ResNet
summary: Deep residual networks that is pre-trained from ImageNet database.
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: resnet.png
featured_image_2: no-image
---

### Model Description

Resnet models were proposed in "Deep Residual Learning for Image Recognition".
Here we have the 5 versions of resnet models, which contains 5, 34, 50, 101, 152 layers repspectively.
Detailed model architectures can be found in Table 1.
Their 1-crop error rates on imagenet dataset with pretrained models are list below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  resnet18       | 30.24       | 10.92       |
|  resnet34       | 26.70       | 8.58        |
|  resnet50       | 23.85       | 7.13        |
|  resnet101      | 22.63       | 6.44        |
|  resnet152      | 21.69       | 5.94        |

### Notes on Inputs

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`. You can use the following transform to normalize:

```
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
```

### Example:

```python
import torch
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True)
model = torch.hub.load('pytorch/vision', 'resnet152', pretrained=True)
```

### Resources:

 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
