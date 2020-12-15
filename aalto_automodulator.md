---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Automodulator
summary: Generative autoencoder for scale-specific fusion of multiple input face images.
category: researchers
image: automodulator1.png
author: Ari Heljakka
tags: [vision, generative]
github-link: https://github.com/AaltoVision/automodulator/tree/hub
github-id: AaltoVision/automodulator
featured_image_1: automodulator2.png
featured_image_2: no-image
accelerator: cuda
order: 10
---
<!-- REQUIRED: provide a working script to demonstrate it works with torch.hub, example below -->

```python
import torch
model = torch.hub.load('AaltoVision/automodulator:hub', 'ffhq512', pretrained=True, force_reload=True, source='github')
model.eval(useLN=False)
```

<!-- Walkthrough a small example of using your model. Ideally, less than 25 lines of code -->

Loads the automodulator [1] model for 512x512 faces (trained on FFHQ [2]).

Scale-specific mixing of multiple real input images is now a breeze, see below.

For the basic workflow, you load in N images and encode them into ``[N,512]`` latent vector with ``model.encode(imgs)``.
To sanity check, you can reconstruct them back into images by ``model.decode(zz)`` where ``zz`` can be a single-image latent or
an instance of ``model.zbuilder()`` which can mix the original latents in arbitrary ways.

```python

# Preliminaries

import sys
sys.argv = ['none'] # For Jupyter/Colab only
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import urllib
from PIL import Image

def show(img):
  nrow = max(2, (len(img)+1)//2)
  ncol = min(2, (len(img)+1)//2)
  img = make_grid(img, nrow=nrow, scale_each=True, normalize=True)
  fig = plt.figure(figsize=(4*nrow,4*ncol))
  plt.imshow(img.permute(1, 2, 0).cpu().numpy())  
```

Load images and reconstruct (replace URLs with your own):

```python
simg = ['https://github.com/AaltoVision/automodulator/raw/hub/fig/source-0.png',
        'https://github.com/AaltoVision/automodulator/raw/hub/fig/source-1.png']

imgs = torch.stack([model.tf()(Image.open(urllib.request.urlopen(simg[0]))),
                    model.tf()(Image.open(urllib.request.urlopen(simg[1])))]).to('cuda')

z = model.encode(imgs)

omgs = model.decode(z).clamp(min=-1, max=1)
# OR: omgs = model.reconstruct(imgs).clamp(min=-1, max=1)

for (i,o) in zip(imgs,omgs):
  show([i, o])
```
Start mixing. For instance, drive the coarse features (4x4 to 8x8) of the bottom-left image BY the top-right:

```python
mixed = model.decode(model.zbuilder().hi(z[1])
                                     .mid(z[0])
                                     .lo(z[0]))

# Equivalent to: model.zbuilder().use(z[1],[0,2]).use(z[0],[2,5]).use(z[0],[5,-1]))

show([torch.ones_like(imgs[0]), imgs[1], imgs[0], mixed[0]])
```

You can use either the shorthand ```model.zbuilder().hi(z[i])``` etc. or the lower-level ```model.zbuilder().use(z[i], [first_block, last_block])``` where ```last_block = -1``` denotes the rest of the blocks.

You can also do random sampling:
```python
random_imgs = model.generate(2).cpu()
show(random_imgs[0].unsqueeze(0))
```

Or, you can generate random samples conditioned on the specific-scale features of your input image:

```python
mixed = model.decode(model.zbuilder(batch_size=6).mid(z[1]).lo(z[1]))
show(mixed)
```

For unsupervised attribute manipulation with features more specific than what you get with style mixing, you can use exemplars of your own (just find the average differences of latent codes on opposing exemplars)
or you can use the pre-calculated ones (from 2x16 exemplars) as below. The larger the exemplar set, the better the attribute vector, but you can get by with as little as two opposing sets of 1 to 4 exemplars each.

```python
urllib.request.urlretrieve('https://github.com/AaltoVision/automodulator/raw/master/pioneer/attrib/smile_delta512-16', 'smile_delta512-16')
smile_delta = torch.load('smile_delta512-16')
#OR: my_attribute_delta = (model.encode(imgs_with_attr) - model.encode(imgs_without_attr)).mean(0) # yields 512-d latent (difference) vector

z_add_smile = z[0].unsqueeze(0) + 0.5*smile_delta
z_no_smile = z[0].unsqueeze(0) - 1.5*smile_delta
mod = model.decode(model.zbuilder().hi(z[0]).mid(z_no_smile).lo(z[0]))
show([imgs[0], mod[0]])
```

<!-- REQUIRED: detailed model description below, in markdown format, feel free to add new sections as necessary -->
### Model Description

The model incorporates the encoder-decoder architecture of [Deep Automodulators](https://arxiv.org/abs/1912.10321) [1] trained on FFHQ [2] up to 512x512.
It allows for instant style mixing of real input images, as well as generating random samples such that their properties on certain scales are fixed on a specific input image.
Input images are expected to be centered and aligned as in FFHQ ([script](https://gist.github.com/heljakka/7163e9f735174bbcdd103c4c13396952)). The ```model.tf()``` then provides the sufficient pre-inference transformations.

The model 'ffhq512' is recommended for all face data modification tasks. The other models in the paper have only been optimized for random sampling.


### References

[1] Heljakka, A., Hou, Y., Kannala, J., and Solin, A. (2020). **Deep Automodulators**. In *Advances in Neural Information Processing Systems (NeurIPS)*. [[arXiv preprint]](https://arxiv.org/abs/1912.10321).

[2] Karras, T., Laine, S., and Aila, T. (2019). **A style-based generator architecture for generative adversarial networks.** In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4401–4410, 2019.
