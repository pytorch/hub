---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
<!-- Only change fields below -->
title: <REQUIRED: short model name>
summary: <REQUIRED: 1-2 sentences>
image: <REQUIRED: best image to represent your model>
author: <REQUIRED>
tags: <REQUIRED: [tag1, tag2, ...]>
github-link: <REQUIRED>
featured_image_1: <OPTIONAL: use no-image if not applicable>
featured_image_2: <OPTIONAL: use no-image if not applicable>
---
<!-- REQUIRED: provide a working script to demonstrate it works with torch.hub, example below -->
```python
import torch
torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
```

<!-- REQUIRED: detailed model description below, in markdown format, feel free to add new sections as necessary -->
### Model Description

<!-- OPTIONAL: put special requirement of your model here, e.g. only supports Python3 -->
### Requiresments

<!-- OPTIONAL: put link to referece papers -->
### References

