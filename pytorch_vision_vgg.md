---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: VGG
summary: Award winning models in ILSVRC challenge 2014.
category: researchers
image: pytorch-logo.png
author: Pytorch Team
tags: [CV, image classification]
github-link: https://github.com/pytorch/vision.git
featured_image_1: vgg.png
featured_image_2: no-image
---

### Model Description

Here we have implementations for the models proposed in [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556),
for each configurations and their with bachnorm version.

For example, configuration `A` presented in the paper is `vgg11`, configuration `B` is `vgg13`, configuration `D` is `vgg16`
and configuration `E` is `vgg19`. Their batchnorm version are suffixed with `_bn`.

Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  vgg11          | 30.98       | 11.37       |
|  vgg11_bn       | 26.70       | 8.58        |
|  vgg13          | 30.07       | 10.75       |
|  vgg13_bn       | 28.45       | 9.63        |
|  vgg16          | 28.41       | 9.62        |
|  vgg16_bn       | 26.63       | 8.50        |
|  vgg19          | 27.62       | 9.12        |
|  vgg19_bn       | 25.76       | 8.15        |

### Notes on Inputs

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

### Example

```python
import torch
model = torch.hub.load('pytorch/vision', 'vgg11', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg11_bn', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg13', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg13_bn', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg16_bn', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg19', pretrained=True)
model = torch.hub.load('pytorch/vision', 'vgg19_bn', pretrained=True)
```

### Resources

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
