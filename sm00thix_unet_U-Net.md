---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: U-Net
summary: U-Net implementation with options for number of input/output channels, padding, Batch/Layer Normalization, and bilinear/TransConv upsampling.
image: unet_diagram.png
author: Ole-Christian Galbo Engstrøm
tags: [vision, scriptable]
github-link: https://github.com/sm00thix/unet/blob/main/unet.py
github-id: sm00thix/unet
featured_image_1: unet_fat_content_regression.png
featured_image_2: no-image
accelerator: "cuda-optional"
---

```python
import torch

# These are the default parameters. They are written out for clarity. Currently no pretrained weights are available.
model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False, in_channels=3, out_channels=1, pad=True, bilinear=True, normalization=None, depth=5)
# or
# model = torch.hub.load('sm00thix/unet', 'unet_bn', **kwargs) # Convenience function equivalent to torch.hub.load('sm00thix/unet', 'unet', normalization='bn' **kwargs)
# or
# model = torch.hub.load('sm00thix/unet', 'unet_ln', **kwargs) # Convenience function equivalent to torch.hub.load('sm00thix/unet', 'unet', normalization='ln', **kwargs)
# or
# model = torch.hub.load('sm00thix/unet', 'unet_medical', **kwargs) # Convenience function equivalent to torch.hub.load('sm00thix/unet', 'unet', in_channels=1, out_channels=1, **kwargs)
# or
# model = torch.hub.load('sm00thix/unet', 'unet_transconv', **kwargs) # Convenience function equivalent to torch.hub.load('sm00thix/unet', 'unet', bilinear=False, **kwargs)
```

### Model Description
This is an implementation of U-Net [[1]](#references). It comes with the following options for customization.

1. Number of input and output channels

    `in_channels` is the number of channels in the input image.
    `out_channels` is the number of channels in the output image.
2. Upsampling
    1. `bilinear = False`: Transposed convolution with a 2x2 kernel applied with stride 2. This is followed by a ReLU.
    2. `bilinear = True`: Factor 2 bilinear upsampling followed by convolution with a 1x1 kernel applied with stride 1.
3. Padding
    1. `pad = True`: The input size is retained in the output by zero-padding convolutions and, if necessary, the results of the upsampling operations.
    2. `pad = False`: The output is smaller than the input as in the original implementation. In this case, every 3x3 convolution layer reduces the height and width by 2 pixels each. Consequently, the right side of the U-Net has a smaller spatial size than the left size. Therefore, before concatenating, the central slice of the left tensor is cropped along the spatial dimensions to match those of the right tensor.
4. Normalization following the ReLU which follows each convolution and transposed convolution.
    1. `normalization = None`: Applies no normalization.
    2. `normalization = "bn"`: Applies batch normalization [[2]](#references).
    3. `normalization = "ln"`: Applies layer normalization [[3]](#references). A permutation of dimensions is performed before the layer to ensure normalization is applied over the channel dimension. Afterward, the dimensions are permuted back to their original order.
5. Depth

    `depth` is the The depth of the U-Net. This is the number of steps in the encoder and decoder paths. This is one less than the number of downsampling and upsampling blocks. The number of intermediate channels is 64*2**`depth`, i.e. [64, 128, 256, 512, 1024] for `depth` = 5.

In particular, setting `bilinear = False`, `pad = False`, `normalization = None`, and `depth = 5` will yield the U-Net as originally designed. Generally, however, `bilinear = True` is recommended to avoid checkerboard artifacts.

As in the original implementation, all weights are initialized by sampling from a Kaiming He Normal Distribution [[4]](#references), and all biases are initialized to zero. If Batch Normalization or Layer Normalization is used, the weights of those layers are initialized to one and their biases to zero.

### References
If you use this U-Net implementation, please cite Engstrøm et al. [[5]](#references) who developed this implementation as part of their work on chemical map geenration of fat content in images of pork bellies.
1. [O. Ronneberger, P. Fischer, and Thomas Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.](https://arxiv.org/abs/1505.04597)
2. [S. Ioffe and C. Szegedy (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*.](https://arxiv.org/abs/1502.03167)
3. [J. L. Ba and J. R. Kiros and G. E. Hinton (2016). Layer Normalization.](https://arxiv.org/abs/1607.06450)
4. [K. He and X. Zhang and S. Ren and J. Sun (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
5. [O.-C. G. Engstrøm and M. Albano-Gaglio and E. S. Dreier and Y. Bouzembrak and M. Font-i-Furnols and P. Mishra and K. S. Pedersen (2025). Transforming Hyperspectral Images Into Chemical Maps: A Novel End-to-End Deep Learning Approach.](https://arxiv.org/abs/2504.14131)