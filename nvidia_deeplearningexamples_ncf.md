---
layout: hub_detail
background-class: hub-background
body-class: hub
title: NCF
summary: Neural Collaborative Filtering model for providing recommendations
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [nlp]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF
featured_image_1: ncf_diagram.png
featured_image_2: no-image
---

```python
import torch
hub_model = torch.hub.load('nvidia/DeepLearningExamples', 'nvidia_ncf', pretrained=False, nb_users=100, nb_items=100)
```

will create an NCF model with 100 users and 100 items. For more information on how to train it, visit: [github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) and/or [NGC](https://ngc.nvidia.com/catalog/model-scripts/nvidia:ncf_for_pytorch)

To play with a model pre-trained on [ml-20m dataset](https://grouplens.org/datasets/movielens/20m/), run:
```python
import torch
hub_model = torch.hub.load('nvidia/DeepLearningExamples', 'nvidia_ncf', pretrained=True)
```

### Model Description

The NCF model focuses on providing recommendations, also known as collaborative filtering; with implicit feedback. The training data for this model should contain binary information about whether a user interacted with a specific item.
NCF was first described by Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua in the [Neural Collaborative Filtering paper](https://arxiv.org/abs/1708.05031).

This implementation focuses on the NeuMF instantiation of the NCF architecture.
We modified it to use dropout in the FullyConnected layers. This reduces overfitting and increases the final accuracy.

### Example

Here's a sample execution on a dummy input (3 users, 3 items):

```python
import torch
print('\nLoading NCF model from torch.hub.')
hub_model = torch.hub.load('nvidia/DeepLearningExamples', 'nvidia_ncf', pretrained=True)
hub_model = hub_model.cuda()
hub_model.eval()
input_users=torch.tensor([0,1,2]).cuda()
input_items=torch.tensor([0,1,2]).cuda()
with torch.no_grad():
    out = hub_model.forward(input_users, input_items, sigmoid=True)
print('\nNCF model test output:')
print(out.size())
```
### Details
For detailed information on model input and output, training recipies, inference and performance visit: [github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) and/or [NGC](https://ngc.nvidia.com/catalog/model-scripts/nvidia:ncf_for_pytorch)

### References

 - [Neural Collaborative Filtering paper](https://arxiv.org/abs/1708.05031)
 - [NCF on NGC](https://ngc.nvidia.com/catalog/model-scripts/nvidia:ncf_for_pytorch)
 - [NCF on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF)
