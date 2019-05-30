---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: Inception_v3
summary: 1st Runner Up for image classification in ILSVRC (ImageNet Large Scale Visual Recognition Competition) 2015.
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: inception_v3.png
featured_image_2: no-image
---

### Model Description

Model `inception_v3` is from the [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) paper

The 1-crop error rates on the imagenet dataset with the pretrained model are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  inception_v3        | 22.55       | 6.44        |

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
model = torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)
```

### Resources:

 - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).
