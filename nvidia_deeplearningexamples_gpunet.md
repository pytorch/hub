---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GPUNet
summary: GPUNet is a new family of Convolutional Neural Networks designed to max out the performance of NVIDIA GPU and TensorRT.
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [vision]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/GPUNet
github-id: NVIDIA/DeepLearningExamples
featured_image_1: classification.jpg
featured_image_2: no-image
accelerator: cuda
---


### Model Description
GPUNets are a new family of deployment and production ready Convolutional Neural Networks from NVIDIA auto-designed to max out the performance of NVIDIA GPU and TensorRT. 

Crafted by NVIDIA AI using novel Neural Architecture Search(NAS) methods, GPUNet demonstrates state-of-the-art inference performance up to 2x faster than EfficientNet-X and FBNet-V3. This notebook allows you to load and test all the the GPUNet model implementation listed in our [CVPR-2022 paper](https://arxiv.org/pdf/2205.00841.pdf). You can use this notebook to quickly load each one of listed models to perform inference runs.

### Example
In the example below the pretrained ***GPUNet-0*** model is loaded by default to perform inference on image and present the result. You can switch the default pre-trained model loading from GPUNet-0 to one of these: GPUNet-1, GPUNet-2, GPUNet-P0, GPUNet-P1, GPUNet-D1 or GPUNet-D2.
### Install pre-requisites
To run the example you need some extra python packages installed. These are needed for preprocessing images and visualization.
```python
!pip install validators matplotlib
!pip install timm==0.5.4
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

device = torch.device("cuda") 
print(f'Using {device} for inference')

if torch.cuda.is_available():
    !nvidia-smi
else:
    torch.device("cpu")
```

### Load Pretrained model
Loads NVIDIA GPUNet-0 model by default pre-trained on IMAGENET dataset. You can switch the default pre-trained model loading from GPUNet-0 to one of the following models listed below. 

The model architecture is visible as output of the loaded model. For details architecture and latency info please refer to Figure#[6](https://drive.google.com/file/d/1pLA0OopYVRpYreIhQQKPdNauaaBEGITe/view) and Table#[3](https://drive.google.com/file/d/1pLA0OopYVRpYreIhQQKPdNauaaBEGITe/view) in the CVPR-2022 paper. 

Please pick and choose one of the following pre-trained models:

| TorchHub model | Description |
| :----- | :----- |
| `GPUNet-0` | GPUNet-0 has the fastest measured latency on GV100 |
| `GPUNet-1` | GPUNet-1 has improved accuracy with one additional layer on GPUNet-0|
| `GPUNet-2` | GPUNet-2 has higher accuracy with two additional layers on GPUNet-0 |
| `GPUNet-P0` | GPUNet-P0 is the distilled model with higher accuracy than GPUNet-0 but similar latency|
| `GPUNet-P1` | GPUNet-P1 is distilled model with even higher accuracy than GPUNet-1 but similar latency |
| `GPUNet-D1` | GPUNet-D1 has the second highest accuracy amongst all GPUNets|
| `GPUNet-D2` | GPUNet-D2 has the highest accuracy amongst all GPUNets |

***Note***: Loaded model defaults to fp32 weights. To switch to fp16 gpunet model weights, set parameter ***model_math='fp16'***.
```python
gpunet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type='GPUNet-0')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

gpunet.eval().to(device)

```

### Prepare inference data
Prepare sample input data for inference.
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

print("Ready to run inference...")
```

### Run inference
Use `pick_n_best(predictions=output, n=topN)` helper function to pick N most probable hypotheses according to the model.

***Note***: If using fp16 gpunet model weights replace gpunet(batch) with ***gpunet(batch.half())***.
```python
with torch.no_grad():
    output = torch.nn.functional.softmax(gpunet(batch), dim=1)
    
results = utils.pick_n_best(predictions=output, n=5)
```

### Display result
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
[github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/GPUNets)

### References

 - [GPUNets: Searching Deployable Convolution Neural Networks for GPUs](https://arxiv.org/pdf/2205.00841.pdf)
 - [model on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/GPUNet)
 - [pretrained model on NGC (GPUNet-0)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/gpunet_0_pyt_ckpt)
 - [pretrained model on NGC (GPUNet-1)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/gpunet_1_pyt_ckpt)
 - [pretrained model on NGC (GPUNet-2)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/gpunet_2_pyt_ckpt)
 - [pretrained distilled model on NGC (GPUNet-P0)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/gpunet_p0_pyt_ckpt)
 - [pretrained, distilled model on NGC (GPUNet-P1)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/gpunet_p1_pyt_ckpt)
 - [pretrained, distilled model on NGC (GPUNet-D1)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/gpunet_d1_pyt_ckpt)
 - [pretrained, distilled model on NGC (GPUNet-D2)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/gpunet_d2_pyt_ckpt)