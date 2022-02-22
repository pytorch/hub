---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Semi-supervised and semi-weakly supervised ImageNet Models
summary: ResNet and ResNext models introduced in the "Billion scale semi-supervised learning for image classification" paper
category: researchers
image: ssl-image.png
author: Facebook AI
tags: [vision]
github-link: https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py
github-id: facebookresearch/semi-supervised-ImageNet1K-models
featured_image_1: ssl-image.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/semi-supervised-ImageNet1K-models
---

```python
import torch

# === SEMI-WEAKLY SUPERVISED MODELSP RETRAINED WITH 940 HASHTAGGED PUBLIC CONTENT ===
model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
# ================= SEMI-SUPERVISED MODELS PRETRAINED WITH YFCC100M ==================
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_ssl')
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
print(torch.nn.functional.softmax(output[0], dim=0))

```

### Model Description
This project includes the semi-supervised and semi-weakly supervised ImageNet models introduced in "Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>.

"Semi-supervised" (SSL) ImageNet models are pre-trained on a subset of unlabeled YFCC100M public image dataset and fine-tuned with the ImageNet1K training dataset, as described by the semi-supervised training framework in the paper mentioned above. In this case, the high capacity teacher model was trained only with labeled examples.

"Semi-weakly" supervised (SWSL) ImageNet models are pre-trained on **940 million** public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning on ImageNet1K dataset. In this case, the associated hashtags are only used for building a better teacher model. During training the student model, those hashtags are ingored and the student model is pretrained with a subset of 64M images selected by the teacher model from the same 940 million public image dataset.

Semi-weakly supervised ResNet and ResNext models provided in the table below significantly improve the top-1 accuracy on the ImageNet validation set compared to training from scratch or other training mechanisms introduced in the literature as of September 2019. For example, **We achieve state-of-the-art accuracy of 81.2% on ImageNet for the widely used/adopted ResNet-50 model architecture**.


| Architecture       |   Supervision   | #Parameters | FLOPS | Top-1 Acc. | Top-5 Acc. |
| ------------------ | :--------------:|:----------: | :---: | :--------: | :--------: |
| ResNet-18          | semi-supervised        |14M     | 2B   |     72.8      | 91.5    |
| ResNet-50          | semi-supervised        |25M     | 4B   |     79.3      | 94.9    |
| ResNeXt-50 32x4d   | semi-supervised        |25M     | 4B   |     80.3      | 95.4    |
| ResNeXt-101 32x4d  | semi-supervised        |42M     | 8B   |     81.0      | 95.7    |
| ResNeXt-101 32x8d  | semi-supervised        |88M     | 16B   |     81.7    |  96.1   |
| ResNeXt-101 32x16d | semi-supervised        |193M    | 36B   |     81.9   | 96.2     |
| ResNet-18          | semi-weakly supervised |14M     | 2B   |    **73.4**    |  91.9      |
| ResNet-50          | semi-weakly supervised |25M     | 4B   |    **81.2**    |  96.0      |
| ResNeXt-50 32x4d   | semi-weakly supervised |25M     | 4B   |    **82.2**    |  96.3      |
| ResNeXt-101 32x4d  | semi-weakly supervised |42M     | 8B   |    **83.4**    |  96.8      |
| ResNeXt-101 32x8d  | semi-weakly supervised |88M     | 16B   |  **84.3**    |  97.2    |
| ResNeXt-101 32x16d | semi-weakly supervised |193M    | 36B   |  **84.8**    |  97.4    |


## Citation

If you use the models released in this repository, please cite the following publication (https://arxiv.org/abs/1905.00546).
```
@misc{yalniz2019billionscale,
    title={Billion-scale semi-supervised learning for image classification},
    author={I. Zeki Yalniz and Hervé Jégou and Kan Chen and Manohar Paluri and Dhruv Mahajan},
    year={2019},
    eprint={1905.00546},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
