---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
<!-- Only change fields below(remove this line before submitting a PR). Take inspiration e.g. from pytorch_vision_fcn_resnet101.md -->
title: Once for All: Train One Network and Specialize it for Efficient Deployment
summary: We propose to train a once-for-all (OFA) network that supports diverse architectural settings by decoupling training and search, thus achieving efficient inference across many devices and resource constraints, especially on edge devices.
image: once-for-all.png
author: MIT Han Lab
tags: [vision]
github-link: https://github.com/mit-han-lab/once-for-all
github-id: https://github.com/mit-han-lab
featured_image_1: once-for-all.png
featured_image_2: no-image
accelerator: cuda-optional
---


<!-- REQUIRED: provide a working script to demonstrate it works with torch.hub, example below -->
```python
import torch
model = torch.hub.load('mit-han-lab/once-for-all', 'ofa_supernet_resnet50', pretrained=True).eval()
```

<!-- Walkthrough a small example of using your model. Ideally, less than 25 lines of code -->

<!-- REQUIRED: detailed model description below, in markdown format, feel free to add new sections as necessary -->
### Model Description


<!-- OPTIONAL: put link to reference papers -->
### References
