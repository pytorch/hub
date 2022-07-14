---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: ntsnet
summary: classify birds using this fine-grained image classifier
image: Cub200Dataset.png
author: Moreno Caraffini and Nicola Landro
tags: [vision]
github-link: https://github.com/nicolalandro/ntsnet-cub200
github-id: nicolalandro/ntsnet-cub200
featured_image_1: nts-net.png
featured_image_2: no-image
accelerator: "cuda-optional"
demo-model-link: https://huggingface.co/spaces/pytorch/NTSNET
---

```python
import torch
model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device':'cpu', 'num_classes': 200})
```

### Example Usage

```python
from torchvision import transforms
import torch
import urllib
from PIL import Image

transform_test = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((448, 448)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
model.eval()

url = 'https://raw.githubusercontent.com/nicolalandro/ntsnet-cub200/master/images/nts-net.png'
img = Image.open(urllib.request.urlopen(url))
scaled_img = transform_test(img)
torch_images = scaled_img.unsqueeze(0)

with torch.no_grad():
    top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(torch_images)

    _, predict = torch.max(concat_logits, 1)
    pred_id = predict.item()
    print('bird class:', model.bird_classes[pred_id])
```

### Model Description
This is an nts-net pretrained with CUB200 2011 dataset, which is a fine grained dataset of birds species.

### References
You can read the full paper at this [link](http://artelab.dista.uninsubria.it/res/research/papers/2019/2019-IVCNZ-Nawaz-Birds.pdf).
```bibtex
@INPROCEEDINGS{Gallo:2019:IVCNZ,
  author={Nawaz, Shah and Calefati, Alessandro and Caraffini, Moreno and Landro, Nicola and Gallo, Ignazio},
  booktitle={2019 International Conference on Image and Vision Computing New Zealand (IVCNZ 2019)},
  title={Are These Birds Similar: Learning Branched Networks for Fine-grained Representations},
  year={2019},
  month={Dec},
}
```
