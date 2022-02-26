---
layout: hub_detail
background-class: hub-background
body-class: hub
title: SE-ResNeXt101
summary: ResNeXt with Squeeze-and-Excitation module added, trained with mixed precision using Tensor Cores.
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [vision]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d
github-id: NVIDIA/DeepLearningExamples
featured_image_1: SEArch.png
featured_image_2: classification.jpg
accelerator: cuda
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/SE-ResNeXt101
---


### Model Description

The ***SE-ResNeXt101-32x4d*** is a [ResNeXt101-32x4d](https://arxiv.org/pdf/1611.05431.pdf)
model with added Squeeze-and-Excitation module introduced
in the [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) paper.

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 3x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

We use [NHWC data layout](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) when training using Mixed Precision.

#### Model architecture

![SEArch](https://pytorch.org/assets/images/SEArch.png)

_Image source: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)_

Image shows the architecture of SE block and where is it placed in ResNet bottleneck block.


Note that the SE-ResNeXt101-32x4d model can be deployed for inference on the [NVIDIA Triton Inference Server](https://github.com/NVIDIA/trtis-inference-server) using TorchScript, ONNX Runtime or TensorRT as an execution backend. For details check [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se_resnext_for_triton_from_pytorch).

### Example

In the example below we will use the pretrained ***SE-ResNeXt101-32x4d*** model to perform inference on images and present the result.

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
```python
resneXt = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

resneXt.eval().to(device)
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
    output = torch.nn.functional.softmax(resneXt(batch), dim=1)
    
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
[github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d)
and/or [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se_resnext_for_pytorch).


### References

 - [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
 - [model on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d)
 - [model on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se_resnext_for_pytorch)
 - [pretrained model on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/seresnext101_32x4d_pyt_amp)
