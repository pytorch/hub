---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: squeezenet
summary: Alexnet-level accuracy with 50x fewer parameters.
category: research
image: pytorch-logo.png
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: squeezenet.png
featured_image_2:
---

### Model Description

Model `squeezenet1_0` is from the [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf) paper

Model `squeezenet1_1` is from the [official squeezenet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1).
It has 2.4x less computation and slightly fewer parameters than `squeezenet1_0`, without sacrificing accuracy.

Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  squeezenet1_0  | 41.90       | 19.58       |
|  squeezenet1_1  | 41.81       | 19.38       |

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
model = torch.hub.load('pytorch/vision', 'squeezenet1_0', pretrained=True)
model = torch.hub.load('pytorch/vision', 'squeezenet1_1', pretrained=True)
```

### Resources:

 - [Squeezenet: Alexnet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf).
