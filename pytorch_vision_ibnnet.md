---
layout: hub_detail
background-class: hub-background
body-class: hub
title: IBN-Net
summary: Networks with domain/appearance invariance
category: researchers
image: ibnnet.png
author: Xingang Pan
tags: [vision]
github-link: https://github.com/XingangPan/IBN-Net
github-id: XingangPan/IBN-Net
featured_image_1: ibnnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/IBN-Net
---

```python
import torch
model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
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

IBN-Net is a CNN model with domain/appearance invariance.
Motivated by style transfer works, IBN-Net carefully unifies instance normalization and batch normalization in a single deep network.
It provides a simple way to increase both modeling and generalization capacities without adding model complexity.
IBN-Net is especially suitable for cross domain or person/vehicle re-identification tasks.

The corresponding accuracies on ImageNet dataset with pretrained models are listed below.

| Model name | Top-1 acc   | Top-5 acc   |
| --------------- | ----------- | ----------- |
| resnet50_ibn_a  | 77.46       | 93.68       |
| resnet101_ibn_a | 78.61       | 94.41       |
| resnext101_ibn_a | 79.12      | 94.58       |
| se_resnet101_ibn_a | 78.75    | 94.49       |

The rank1/mAP on two Re-ID benchmarks Market1501 and DukeMTMC-reID are listed below (from [michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)).

| Backbone | Market1501 | DukeMTMC-reID |
| --- | -- | -- |
| ResNet50 | 94.5 (85.9) | 86.4 (76.4) |
| ResNet101 | 94.5 (87.1) |  87.6 (77.6) |
| SeResNet50 | 94.4 (86.3) | 86.4 (76.5) |
| SeResNet101 | 94.6 (87.3) | 87.5 (78.0) |
| SeResNeXt50 | 94.9 (87.6) | 88.0 (78.3) |
| SeResNeXt101 | 95.0 (88.0) | 88.4 (79.0) |
| ResNet50-IBN-a | 95.0 (88.2) | 90.1 (79.1) |

### References

 - [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net](https://arxiv.org/abs/1807.09441)
