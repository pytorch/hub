---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: YOLOv5
summary: YOLOv5 in PyTorch > ONNX > CoreML > TFLite
image: ultralytics_yolov5_img0.jpg
author: Ultralytics LLC
tags: [vision, scriptable]
github-link: https://github.com/ultralytics/yolov5
github-id: ultralytics/yolov5
featured_image_1: ultralytics_yolov5_img1.jpg
featured_image_2: ultralytics_yolov5_img2.png
accelerator: cuda-optional
---

## Before You Start

Start from a working python environment with **Python>=3.8** and **PyTorch>=1.6** installed, as well as `pyyaml>=5.3` for reading YOLOv5 configuration files. To install PyTorch see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). To install dependencies:
```bash
$ pip install -U opencv-python pillow pyyaml tqdm  # install dependencies
```

## Model Description

<img width="800" alt="YOLOv5 Models" src="https://user-images.githubusercontent.com/26833433/97808084-edfcb100-1c64-11eb-83eb-ffed43a0859f.png">
&nbsp;

YOLOv5 is a family of compound-scaled object detection models trained on COCO 2017, and includes built-in functionality for Test Time Augmentation (TTA), Model Ensembling, Rectangular Inference, Hyperparameter Evolution.

| Model | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 37.0     | 37.0     | 56.2     | **2.4ms** | **416** || 7.5M   | 13.2B
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 44.3     | 44.3     | 63.2     | 3.4ms     | 294     || 21.8M  | 39.4B
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 47.7     | 47.7     | 66.5     | 4.4ms     | 227     || 47.8M  | 88.1B
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 49.2 | 49.2 | 67.7 | 6.9ms     | 145     || 89.0M  | 166.4B
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0) + TTA|**50.8**| **50.8** | **68.9** | 25.5ms    | 39      || 89.0M  | 354.3B

<img src="https://user-images.githubusercontent.com/26833433/90187293-6773ba00-dd6e-11ea-8f90-cd94afc0427f.png" width="800">  
** GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.


## Load YOLOv5 From PyTorch Hub

### Example

To load YOLOv5 from PyTorch Hub for inference with PIL, OpenCV, Numpy or PyTorch inputs:
```python
import cv2
import torch
from PIL import Image, ImageDraw

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()  # yolov5s.pt
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

# Images
for f in ['zidane.jpg', 'bus.jpg']:  # download 2 images
    torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)
img1 = Image.open('zidane.jpg')  # PIL image
img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batched list of images

# Inference
with torch.no_grad():
    prediction = model(imgs, size=640)  # includes NMS

print(prediction[0])  # print img1 predictions
#          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
# tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
#         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
#         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
#         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])
```

To print/plot results:
```python
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

## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit [https://www.ultralytics.com](https://www.ultralytics.com) or email Glenn Jocher at [glenn.jocher@ultralytics.com](mailto:glenn.jocher@ultralytics.com). 

&nbsp;
