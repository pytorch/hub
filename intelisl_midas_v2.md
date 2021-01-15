---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: MiDaS
summary: The MiDaS v2.1 model for computing relative depth from a single image.
image: intel-logo.png
author: Intel ISL
tags: [vision]
github-link: https://github.com/intel-isl/MiDaS
github-id: intel-isl/MiDaS
featured_image_1: midas_samples.png
featured_image_2: no-image
accelerator: cuda-optional
---

```python
import torch
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()
```

will load the MiDaS v2.1 model. The model expects 3-channel RGB images of shape ```(3 x H x W)```. Images are expected to be normalized using
`mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`. 
`H` and `W` need to be divisible by `32`. For optimal results `H` and `W` should be close to `384` (the training resolution). 
We provide a custom transformation that performs resizing while maintaining aspect ratio. 

### Model Description

[MiDaS](https://arxiv.org/abs/1907.01341) computes relative inverse depth from a single image. The model has been trained on 10 distinct dataset using 
multi-objective optimization to ensure high quality on a wide range of inputs.


### Example Usage

Download an image from the PyTorch homepage
```python
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
```

Load large or small model

```python
use_large_model = True

if use_large_model:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
else:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
```

Move model to GPU if available
```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
```


Load transforms to resize and normalize the image for large or small model
```python
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if use_large_model:
    transform = midas_transforms.default_transform
else:
    transform = midas_transforms.small_transform
```

Load image and apply transforms
```python
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)
```


Predict and resize to original resolution
```python
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
output = prediction.cpu().numpy()
```

Show result
```python 
plt.imshow(output)
# plt.show()
```

### Reference
[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341)

Please cite our paper if you use our model:
```bibtex
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
