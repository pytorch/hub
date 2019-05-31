---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: Densenet
summary: Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion.
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: densenet1.png
featured_image_2: densenet2.png
---

### Model Description

Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. 

The 1-crop error rates on the imagenet dataset with the pretrained model are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  densenet121        | 25.35       | 7.83        |
|  densenet169        | 24.00       | 7.00        |
|  densenet201        | 22.80       | 6.43        |
|  densenet161        | 22.35       | 6.20        |

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
model = torch.hub.load('pytorch/vision', 'densenet121', pretrained=True)
model = torch.hub.load('pytorch/vision', 'densenet169', pretrained=True)
model = torch.hub.load('pytorch/vision', 'densenet201', pretrained=True)
model = torch.hub.load('pytorch/vision', 'densenet161', pretrained=True)
```

### Resources:

 - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).
