---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: YOLOv5
summary: YOLOv5 in PyTorch > ONNX > CoreML > TFLite
image: images/ultralytics_yolov5_img1.jpg
author: Ultralytics LLC
tags: [vision, scriptable]
github-link: https://github.com/ultralytics/yolov5
github-id: ultralytics/yolov5
featured_image_1: ultralytics_yolov5_img1.jpg
featured_image_2: ultralytics_yolov5_img2.jpg
accelerator: cuda-optional
---

🚀  This guide explains how to load YOLOv5 from PyTorch Hub https://pytorch.org/hub/

## Before You Start

You should start from a working python environment with **Python>=3.8** and **PyTorch>=1.6** installed, as well as `pyyaml>=5.3` for reading YOLOv5 configuration files. To install PyTorch see https://pytorch.org/get-started/locally/. To install dependencies:
```bash
$ pip install -U opencv-python pillow pyyaml tqdm  # install dependencies
```

You do **not** need to clone the https://github.com/ultralytics/yolov5 repository.


## Load YOLOv5 From PyTorch Hub

### Simple Example

This simple example loads a pretrained YOLOv5s model from PyTorch Hub as `model` and passes a PIL image for inference. `'yolov5s'` is the lightest and fastest YOLOv5 model. Other available models are `'yolov5m'`, `'yolov5l'`, and `'yolov5x'`, the largest and most accurate model. For details on all models please see the [README](https://github.com/ultralytics/yolov5#pretrained-checkpoints).
```python
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()  # yolov5s.pt
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

# Images
img = Image.open('zidane.jpg')  # PIL image

# Inference
prediction = model(img)  # includes NMS
```

### Detailed Example

To load YOLOv5 from PyTorch Hub for **batched inference** with PIL, OpenCV, Numpy or PyTorch inputs, with results plotted, saved and printed to screen:
```python
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# Model
model = torch.hub.load(name='yolov5s', pretrained=True, channels=3, classes=80).fuse().eval()  # yolov5s.pt
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

# Images
for f in ['zidane.jpg', 'bus.jpg']:  # download 2 images
    torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)
img1 = Image.open('zidane.jpg')  # PIL image
img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
img3 = np.zeros((640, 1280, 3))  # numpy array
imgs = [img1, img2, img3]  # batched list of images

# Inference
with torch.no_grad():
    prediction = model(imgs, size=640)  # includes NMS

# Plot
for i, (img, pred) in enumerate(zip(imgs, prediction)):
    str = 'Image %g/%g: %gx%g ' % (i + 1, len(imgs), *img.shape[:2])
    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
    if pred is not None:
        for c in pred[:, -1].unique():
            n = (pred[:, -1] == c).sum()  # detections per class
            str += '%g %ss, ' % (n, model.names[int(c)])  # add to string
        for *box, conf, cls in pred:  # xyxy, confidence, class
            label = model.names[int(cls)] if hasattr(model, 'names') else 'class_%g' % cls
            # str += '%s %.2f, ' % (label, conf)  # label
            ImageDraw.Draw(img).rectangle(box, width=3)  # plot
    img.save('results%g.jpg' % i)  # save
    print(str + 'Done.')
```
<img src="https://user-images.githubusercontent.com/26833433/96368758-114c3a00-1156-11eb-9ede-a03f9aff42e4.jpg" width="480">  <img src="https://user-images.githubusercontent.com/26833433/97121810-54dc1080-1721-11eb-8c25-451ccb1452e4.jpg" width="320">

For all inference options see YOLOv5 `autoShape()` forward method:
https://github.com/ultralytics/yolov5/blob/3b57cb56412976847b91ef184ca2f2a99491d8ab/models/common.py#L129-L136

### Vary Input Channels
To load a pretrained YOLOv5s model with 4 input channels rather than the default 3:
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, channels=4)
```
In this case the model will be composed of pretrained weights **except for** the very first input layer, which is no longer the same shape as the pretrained input layer. The input layer will remain initialized by random weights.

### Vary Number of Classes
To load a pretrained YOLOv5s model with 10 output classes rather than the default 80:
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=10)
```
In this case the model will be composed of pretrained weights **except for** the output layers, which are no longer the same shape as the pretrained output layers. The output layers will remain initialized by random weights.


## Pretrained Checkpoints

| Model | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 37.0     | 37.0     | 56.2     | **2.4ms** | **416** || 7.5M   | 13.2B
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 44.3     | 44.3     | 63.2     | 3.4ms     | 294     || 21.8M  | 39.4B
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 47.7     | 47.7     | 66.5     | 4.4ms     | 227     || 47.8M  | 88.1B
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | **49.2** | **49.2** | **67.7** | 6.9ms     | 145     || 89.0M  | 166.4B
| | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0) + TTA|**50.8**| **50.8** | **68.9** | 25.5ms    | 39      || 89.0M  | 354.3B
| | | | | | || |
| [YOLOv3-SPP](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | 45.6     | 45.5     | 65.2     | 4.5ms     | 222     || 63.0M  | 118.0B

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results in the table denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or test-time augmentation. **Reproduce** by `python test.py --data coco.yaml --img 640 --conf 0.001`  
** Speed<sub>GPU</sub> measures end-to-end time per image averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) instance with one V100 GPU, and includes image preprocessing, PyTorch FP16 image inference at --batch-size 32 --img-size 640, postprocessing and NMS. Average NMS time included in this chart is 1-2ms/img.  **Reproduce** by `python test.py --data coco.yaml --img 640 --conf 0.1`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce** by `python test.py --data coco.yaml --img 832 --augment` 

&nbsp;

<img src="https://user-images.githubusercontent.com/26833433/90187293-6773ba00-dd6e-11ea-8f90-cd94afc0427f.png" width="800">  
** GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
* [CoreML, ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab Notebook** with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com. 
