---
layout: hub_detail
background-class: hub-background
body-class: hub
title: EfficientNet
summary: EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, being an order-of-magnitude smaller and faster. Trained with mixed precision using Tensor Cores.
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [vision]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet
github-id: NVIDIA/DeepLearningExamples
featured_image_1: classification.jpg
featured_image_2: no-image
accelerator: cuda
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/EfficientNet
---


### Model Description

EfficientNet is an image classification model family. It was first described in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). This notebook allows you to load and test the EfficientNet-B0, EfficientNet-B4, EfficientNet-WideSE-B0 and, EfficientNet-WideSE-B4 models.

EfficientNet-WideSE models use Squeeze-and-Excitation layers wider than original EfficientNet models, the width of SE module is proportional to the width of Depthwise Separable Convolutions instead of block width.

WideSE models are slightly more accurate than original models.

This model is trained with mixed precision using Tensor Cores on Volta and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results over 2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

We use [NHWC data layout](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) when training using Mixed Precision.

### Example

In the example below we will use the pretrained ***EfficientNet*** model to perform inference on image and present the result.

To run the example you need some extra python packages installed. These are needed for preprocessing images and visualization.
```python
!pip install validators matplotlib
```

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
```

Load the model pretrained on IMAGENET dataset.

You can choose among the following models:

| TorchHub entrypoint | Description |
| :----- | :----- |
| `nvidia_efficientnet_b0` | baseline EfficientNet |
| `nvidia_efficientnet_b4` | scaled EfficientNet|
| `nvidia_efficientnet_widese_b0` | model with Squeeze-and-Excitation layers wider than baseline EfficientNet model |
| `nvidia_efficientnet_widese_b4` | model with Squeeze-and-Excitation layers wider than scaled EfficientNet model |

There are also quantized version of the models, but they require nvidia container. See [quantized models](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet#quantization)
```python
efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

efficientnet.eval().to(device)

```

Prepare sample input data.
```python
uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)
```

Run inference. Use `pick_n_best(predictions=output, n=topN)` helper function to pick N most probable hypotheses according to the model.
```python
with torch.no_grad():
    output = torch.nn.functional.softmax(efficientnet(batch), dim=1)
    
results = utils.pick_n_best(predictions=output, n=5)
```

Display the result.
```python
for uri, result in zip(uris, results):
    img = Image.open(requests.get(uri, stream=True).raw)
    img.thumbnail((256,256), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    print(result)
```

### Details
For detailed information on model input and output, training recipies, inference and performance visit:
[github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)
and/or [NGC](https://ngc.nvidia.com/catalog/resources/nvidia:efficientnet_for_pytorch)

### References

 - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
 - [model on NGC](https://ngc.nvidia.com/catalog/resources/nvidia:efficientnet_for_pytorch)
 - [model on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)
 - [pretrained model on NGC (efficientnet-b0)](https://ngc.nvidia.com/catalog/models/nvidia:efficientnet_b0_pyt_amp)
 - [pretrained model on NGC (efficientnet-b4)](https://ngc.nvidia.com/catalog/models/nvidia:efficientnet_b4_pyt_amp)
 - [pretrained model on NGC (efficientnet-widese-b0)](https://ngc.nvidia.com/catalog/models/nvidia:efficientnet_widese_b0_pyt_amp)
 - [pretrained model on NGC (efficientnet-widese-b4)](https://ngc.nvidia.com/catalog/models/nvidia:efficientnet_widese_b4_pyt_amp)
 - [pretrained, quantized model on NGC (efficientnet-widese-b0)](https://ngc.nvidia.com/catalog/models/nvidia:efficientnet_widese_b0_pyt_amp)
 - [pretrained, quantized model on NGC (efficientnet-widese-b4)](https://ngc.nvidia.com/catalog/models/nvidia:efficientnet_widese_b4_pyt_amp)
 