---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ResNext
summary: Next generation ResNets, more efficient and accurate
category: researchers
image: resnext.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
github-id: pytorch/vision
featured_image_1: resnext.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
# or
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
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
#Download ImageNet labels
!wget https://raw.githubusercontent.com/Tylersuard/hub/master/imagenet_classes.txt
```

```
#Apply labels to the tensor
with open ("imagenet_classes.txt", "r") as myfile:
    data=myfile.read()
    processed = [s.strip() for s in data.splitlines()]
    #print(processed)

probabilities = probabilities.cpu()
top5 = torch.topk(probabilities,5)
top5_list = [a.tolist() for a in top5]
#print(top5_list)

for i in range(5):
  print(processed[top5_list[1][i]],top5_list[0][i])
```

### Model Description

Resnext models were proposed in [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
Here we have the 2 versions of resnet models, which contains 50, 101 layers repspectively.
A comparison in model archetechure between resnet50 and resnext50 can be found in Table 1.
Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

|  Model structure  | Top-1 error | Top-5 error |
| ----------------- | ----------- | ----------- |
|  resnext50_32x4d  | 22.38       | 6.30        |
|  resnext101_32x8d | 20.69       | 5.47        |

### References

 - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
