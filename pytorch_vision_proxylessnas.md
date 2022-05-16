---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ProxylessNAS
summary: Proxylessly specialize CNN architectures for different hardware platforms.
category: researchers
image: proxylessnas.png
author: MIT Han Lab
tags: [vision]
github-link: https://github.com/mit-han-lab/ProxylessNAS
github-id: mit-han-lab/ProxylessNAS
featured_image_1: proxylessnas.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/ProxylessNAS
---

```python
import torch
target_platform = "proxyless_cpu"
# proxyless_gpu, proxyless_mobile, proxyless_mobile14 are also avaliable.
model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=True)
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
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
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
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# Download ImageNet labels
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### Model Description

ProxylessNAS models are from the [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332) paper.

Conventionally, people tend to design *one efficient model* for *all hardware platforms*. But different hardware has different properties, for example, CPU has higher frequency and GPU is better at parallization. Therefore, instead of generalizing, we need to **specialize** CNN architectures for different hardware platforms. As shown in below, with similar accuracy, specialization offers free yet significant performance boost on all three platforms.

| Model structure |  GPU Latency | CPU Latency | Mobile Latency
| --------------- | ----------- | ----------- | ----------- |
|  proxylessnas_gpu     |  **5.1ms**   | 204.9ms | 124ms |
|  proxylessnas_cpu     |  7.4ms   | **138.7ms** | 116ms |
|  proxylessnas_mobile  |  7.2ms   | 164.1ms | **78ms**  |

The corresponding top-1 accuracy with pretrained models are listed below.

| Model structure | Top-1 error |
| --------------- | ----------- |
|  proxylessnas_cpu     |  24.7 |
|  proxylessnas_gpu     |  24.9   |
|  proxylessnas_mobile  |  25.4   |
|  proxylessnas_mobile_14  |  23.3   |

### References

 - [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332).
