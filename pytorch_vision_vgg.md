---
layout: hub_detail
background-class: hub-background
body-class: hub
title: vgg-nets
summary: Award winning ConvNets from 2014 Imagenet ILSVRC challenge
category: researchers
image: vgg.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
github-id: pytorch/vision
featured_image_1: vgg.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19_bn', pretrained=True)
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

### References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
