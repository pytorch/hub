---
layout: hub_detail
background-class: hub-background
body-class: hub
title: HarDNet
summary: Harmonic DenseNet pre-trained on ImageNet
category: researchers
image: hardnet.png
author: PingoLH
tags: [vision]
github-link: https://github.com/PingoLH/Pytorch-HarDNet
github-id: PingoLH/Pytorch-HarDNet
featured_image_1: hardnet.png
featured_image_2: hardnet_blk.png
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/HardNet
---

```python
import torch
model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True)
# or any of these variants
# model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet85', pretrained=True)
# model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68ds', pretrained=True)
# model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet39ds', pretrained=True)
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

Harmonic DenseNet (HarDNet) is a low memory traffic CNN model, which is fast and efficient.
The basic concept is to minimize both computational cost and memory access cost at the same
time, such that the HarDNet models are 35% faster than ResNet running on GPU
comparing to models with the same accuracy (except the two DS models that
were designed for comparing with MobileNet).

Here we have the 4 versions of hardnet models, which contains 39, 68, 85 layers
w/ or w/o Depthwise Separable Conv respectively.
Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  hardnet39ds    | 27.92       | 9.57        |
|  hardnet68ds    | 25.71       | 8.13        |
|  hardnet68      | 23.52       | 6.99        |
|  hardnet85      | 21.96       | 6.11        |

### References

 - [HarDNet: A Low Memory Traffic Network](https://arxiv.org/abs/1909.00948)
