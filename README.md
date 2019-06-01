# hub

## Logistics

We accept submission to Pytorch hub through PR in `pytorch/hub` repo. Once the PR is merged into master here, it will show up on Pytorch website in 24 hrs.

Currently we don't support hosting pretrained weights, users with pretrained weights need to host them properly themselves.

## Steps to submit to Pytorch hub

1. Add a `hubconf.py` in your repo, following the instruction in [torch.hub doc](https://pytorch.org/docs/master/hub.html#publishing-models). Verify it's working correctly by running `torch.hub.load(...)` locally.
2. Create a PR with a new `<repo_owner>_<repo_name>_<title>.md` file. In general we recommend one file per model, models with similar structures like `resnet18, resnet50` should be placed in the same file.
3. Please use the template below to create new PRs.
```
---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
<!-- Only change fields below -->
title: <REQUIRED: short model name>
summary: <REQUIRED: 1-2 sentences>
image: <REQUIRED: use pytorch-logo.png if you don't have a logo>
author: <REQUIRED>
tags: <REQUIRED: [tag1, tag2, ...]>
github-link: <REQUIRED>
featured_image_1: <OPTIONAL: use no-image if not applicable>
featured_image_2: <OPTIONAL: use no-image if not applicable>
---
<!-- REQUIRED: detailed model description below, in markdown format, feel free to add new sections as necessary -->
### Model Description

<!-- REQUIRED: provide a working script to demonstrate it works with torch.hub -->
### Example

<!-- OPTIONAL: put special requirement of your model here, e.g. only supports Python3 -->
### Requiresments

```
where an example script could be as simple as a `hub.load()` example.
```python
import torch
torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
```

4. If you have images, place them in `images/` folder and link them correctly in the `[images/featured_image_1/featured_image_2]` fields above.

