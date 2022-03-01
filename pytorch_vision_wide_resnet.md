---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Wide ResNet
summary: Wide Residual Networks
category: researchers
image: wide_resnet.png
author: Sergey Zagoruyko
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
github-id: pytorch/vision
featured_image_1: wide_resnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/Wide_Resnet
---

```python
import torch
# load WRN-50-2:
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
# or WRN-101-2
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
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

Wide Residual networks simply have increased number of channels compared to ResNet.
Otherwise the architecture is the same. Deeper ImageNet models with bottleneck
block have increased number of channels in the inner 3x3 convolution.

The `wide_resnet50_2` and `wide_resnet101_2` models were trained in FP16 with
mixed precision training using SGD with warm restarts. Checkpoints have weights in
half precision (except batch norm) for smaller size, and can be used in FP32 models too.

| Model structure   | Top-1 error | Top-5 error | # parameters |
| ----------------- | :---------: | :---------: | :----------: |
|  wide_resnet50_2  | 21.49       | 5.91        | 68.9M        |
|  wide_resnet101_2 | 21.16       | 5.72        | 126.9M       |

### References

 - [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
 - [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
 - [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
