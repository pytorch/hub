---
layout: hub_detail
background-class: hub-background
body-class: hub
title: EfficientNet
summary: Memory and Performance Efficient Networks with 8 configurations.
category: researchers
image: mobilenet_v3_1.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
github-id: pytorch/vision
featured_image_1: mobilenet_v3_1.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b0', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b1', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b2', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b3', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b4', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b5', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b6', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b7', pretrained=True)
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

The MobileNet v3 architecture is based some stuff.

|  Model structure          | Top-1 error | Top-5 error |
|  ------------------       | ----------- | ----------- |
|  efficientnet_b0          | 22.30       | 6.46        |
|  efficientnet_b1          | 21.35       | 5.82        |
|  efficientnet_b2          | 19.39       | 4.69        |
|  efficientnet_b3          | 17.99       | 3.94        |
|  efficientnet_b4          | 16.61       | 3.40        |
|  efficientnet_b5          | 16.55       | 3.37        |
|  efficientnet_b6          | 15.99       | 3.08        |
|  efficientnet_b7          | 15.87       | 3.09        |


### References

 - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
