---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GhostNet
summary: Efficient networks by generating more features from cheap operations
category: researchers
image: ghostnet.png
author: Huawei Noah's Ark Lab
tags: [vision, scriptable]
github-link: https://github.com/huawei-noah/ghostnet
github-id: huawei-noah/ghostnet
featured_image_1: ghostnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/GhostNet
---

```python
import torch
model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
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

The GhostNet architecture is based on an Ghost module structure which generate more features from cheap operations. Based on a set of intrinsic feature maps, a series of cheap operations are applied to generate many ghost feature maps that could fully reveal information underlying intrinsic features. Experiments conducted on benchmarks demonstrate that the superiority of GhostNet in terms of speed and accuracy tradeoff.

The corresponding accuracy on ImageNet dataset with pretrained model is listed below.

| Model structure | FLOPs       | Top-1 acc   | Top-5 acc   |
| --------------- | ----------- | ----------- | ----------- |
|  GhostNet 1.0x  | 142M        | 73.98       | 91.46       |


### References

You can read the full paper at this [link](https://arxiv.org/abs/1911.11907).

>@inproceedings{han2019ghostnet,
>    title={GhostNet: More Features from Cheap Operations},
>    author={Kai Han and Yunhe Wang and Qi Tian and Jianyuan Guo and Chunjing Xu and Chang Xu},
>    booktitle={CVPR},
>    year={2020},
>}

