---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: cyclegan
summary: generate pictures as famous artis from photos
image: cyclegan_cezanne.png
author: Nicola Landro
tags: [vision, generative]
github-link: https://github.com/nicolalandro/cyclegan-pretrained
github-id: nicolalandro/cyclegan-pretrained
featured_image_1: cyclegan_cezanne.png
featured_image_2: no-image
accelerator: "cuda-optional"
---

```python
import torch
model = torch.hub.load('nicolalandro/cyclegan-pretrained', 'cyclegan', pretrained='img2cezanne', device='cpu')
print(model.trained_models_list)
```

### Example Usage

```python
from torchvision import transforms
import torch
import urllib
from PIL import Image
from torchvision.utils import save_image


model = torch.hub.load('nicolalandro/cyclegan-pretrained', 'cyclegan', pretrained='img2cezanne', device='cpu')
model.eval()

url = 'https://raw.githubusercontent.com/nicolalandro/cyclegan-pretrained/main/images/scala_madonnina_del_mare.jpeg'
img = Image.open(urllib.request.urlopen(url))
scale_factor = 0.8
shape = [int(x * scale_factor) for x in img.size]

transform = transforms.Compose([
    transforms.Resize(size=(shape[0], shape[1])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

scaled_img = transform(img)
torch_images = scaled_img.unsqueeze(0)

with torch.no_grad():
    fake_A = 0.5 * (model(torch_images).data + 1.0)
    save_image(fake_A, 'cezanne_generated.png')
```

### Model Description
This is a cyclegan, a generative model that is trained on an image style and can trasnform the generic images into images with that particular style.


