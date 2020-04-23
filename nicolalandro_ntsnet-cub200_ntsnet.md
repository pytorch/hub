---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: ntsnet
summary: a fine grane model for image classification.
image: nts-net.png
author: Moreno Carraffini and Nicola Landro
tags: [vision]
github-link: https://github.com/nicolalandro/ntsnet_cub200
featured_image_1: no-image
featured_image_2: no-image
accelerator: "cuda-optional"
---

```python
# Preprocessing image of cube
# transform_train = transforms.Compose([
#     transforms.Resize((600, 600), Image.BILINEAR),
#     transforms.CenterCrop((448, 448)),
#     transforms.RandomHorizontalFlip(),  # solo se train
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
import torch
mdoel = torch.hub.load('nicolalandro/ntsnet_cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(img)
# if you need the output props use "concat_out"
```

### Model Description
This is a nts-net pretrained with CUB200 2011 dataset. A fine grane dataset of birds species. 
It is a particular model and if you want to train it use the follow:

```python
import torch

PROPOSAL_NUM = 6
LR = 0.001
WD = 1e-4

net = torch.hub.load('nicolalandro/ntsnet_cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})

creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)

...

for i, data in enumerate(trainloader):
    img, label = data[0].cuda(), data[1].cuda()
    batch_size = img.size(0)
    raw_optimizer.zero_grad()
    part_optimizer.zero_grad()
    concat_optimizer.zero_grad()
    partcls_optimizer.zero_grad()
    _, _, raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
    part_loss = net.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
    raw_loss = creterion(raw_logits, label)
    concat_loss = creterion(concat_logits, label)
    rank_loss = net.ranking_loss(top_n_prob, part_loss)
    partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

    total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
    total_loss.backward()
    raw_optimizer.step()
    part_optimizer.step()
    concat_optimizer.step()
    partcls_optimizer.step()
```

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