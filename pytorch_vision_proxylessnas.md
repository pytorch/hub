---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ProxylessNAS
summary: Proxylessly specialize CNN architectures for different hardware platforms.
category: researchers
image: squeezenet.png
author: Pytorch Team
tags: [vision]
github-link: https://github.com/mit-han-lab/ProxylessNAS
featured_image_1: proxyless_overview.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

```python
import torch
model = torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_cpu', pretrained=True)
model.eval()
```

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

Here's a sample execution.

```python
# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))

```

### Model Description

Model `squeezenet1_0` is from the [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf) paper

Model `squeezenet1_1` is from the [official squeezenet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1).
It has 2.4x less computation and slightly fewer parameters than `squeezenet1_0`, without sacrificing accuracy.

The corresponding top-1 accuracy and speed with pretrained models are listed below.

| Model structure | Top-1 error | 
| --------------- | ----------- | 
|  proxylessnas_cpu     |  24.7 | 
|  proxylessnas_gpu     |  24.9   |
|  proxylessnas_mobile  |  25.4   |
|  proxylessnas_mobile_14  |  23.3   |


The inference speed on various platforms with provided models are given below.


| Model structure |  GPU Latency | CPU Latency | Mobile Latency
| --------------- | ----------- | ----------- | ----------- | 
|  proxylessnas_gpu     |  **5.1ms**   | 204.9ms | 124ms |
|  proxylessnas_cpu     |  7.4ms   | **138.7ms** | 116ms | 
|  proxylessnas_mobile  |  7.2ms   | 164.1ms | **78ms**  |


As shown in above, with similar accuracy, specialization leads to significant efficiency boost.

### References

 - [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332).
