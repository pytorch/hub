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

Start from a **Python>=3.8** environment with **PyTorch>=1.7** installed. To install PyTorch see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). To install YOLOv5 dependencies:
```bash
pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies
```


## Model Description

<img width="800" alt="YOLOv5 Models" src="https://user-images.githubusercontent.com/26833433/103595982-ab986000-4eb1-11eb-8c57-4726261b0a88.png">
&nbsp;

YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite.

| Model | size | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>V100</sub> | FPS<sub>V100</sub> || params | GFLOPS |
|---------- |------ |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases)    |640 |36.8     |36.8     |55.6     |**2.2ms** |**455** ||7.3M   |17.0
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases)    |640 |44.5     |44.5     |63.1     |2.9ms     |345     ||21.4M  |51.3
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases)    |640 |48.1     |48.1     |66.4     |3.8ms     |264     ||47.0M  |115.4
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases)    |640 |**50.1** |**50.1** |**68.7** |6.0ms     |167     ||87.7M  |218.8
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases) + TTA |832 |**51.9** |**51.9** |**69.6** |24.9ms |40      ||87.7M  |1005.3

<img width="800" alt="YOLOv5 Performance" src="https://user-images.githubusercontent.com/26833433/103594689-455e0e00-4eae-11eb-9cdf-7d753e2ceeeb.png">
** GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.


## Load From PyTorch Hub

This simple example loads a pretrained **YOLOv5s** model from PyTorch Hub as `model` and passes two **image URLs** for batched inference. 

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batched list of images

# Inference
results = model(imgs)

# Results
results.print()  
results.save()  # or .show()

# Data
print(results.xyxy[0])  # print img1 predictions (pixels)
#                   x1           y1           x2           y2   confidence        class
# tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
#         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
#         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
```

For YOLOv5 PyTorch Hub inference with **PIL**, **OpenCV**, **Numpy** or **PyTorch** inputs please see the full [YOLOv5 PyTorch Hub Tutorial](https://github.com/ultralytics/yolov5/issues/36).


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit [https://www.ultralytics.com](https://www.ultralytics.com) or email Glenn Jocher at [glenn.jocher@ultralytics.com](mailto:glenn.jocher@ultralytics.com). 

&nbsp;
