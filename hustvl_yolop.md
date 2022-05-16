---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: YOLOP
summary: YOLOP pretrained on the BDD100K dataset
image: yolop.png
author: Hust Visual Learning Team
tags: [vision]
github-link: https://github.com/hustvl/YOLOP
github-id: hustvl/YOLOP
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/YOLOP
---
## Before You Start
To install YOLOP dependencies:
```bash
pip install -qr https://github.com/hustvl/YOLOP/blob/main/requirements.txt  # install dependencies
```


## YOLOP: You Only Look Once for Panoptic driving Perception

### Model Description

<img width="800" alt="YOLOP Model" src="https://github.com/hustvl/YOLOP/raw/main/pictures/yolop.png">
&nbsp;

- YOLOP is an efficient multi-task network that can jointly handle three crucial tasks in autonomous driving: object detection, drivable area segmentation and lane detection. And it is also the first to reach real-time on embedded devices while maintaining state-of-the-art level performance on the **BDD100K** dataset.


### Results

#### Traffic Object Detection Result

| Model          | Recall(%) | mAP50(%) | Speed(fps) |
| -------------- | --------- | -------- | ---------- |
| `Multinet`     | 81.3      | 60.2     | 8.6        |
| `DLT-Net`      | 89.4      | 68.4     | 9.3        |
| `Faster R-CNN` | 77.2      | 55.6     | 5.3        |
| `YOLOv5s`      | 86.8      | 77.2     | 82         |
| `YOLOP(ours)`  | 89.2      | 76.5     | 41         |

#### Drivable Area Segmentation Result

| Model         | mIOU(%) | Speed(fps) |
| ------------- | ------- | ---------- |
| `Multinet`    | 71.6    | 8.6        |
| `DLT-Net`     | 71.3    | 9.3        |
| `PSPNet`      | 89.6    | 11.1       |
| `YOLOP(ours)` | 91.5    | 41         |

#### Lane Detection Result

| Model         | mIOU(%) | IOU(%) |
| ------------- | ------- | ------ |
| `ENet`        | 34.12   | 14.64  |
| `SCNN`        | 35.79   | 15.84  |
| `ENet-SAD`    | 36.56   | 16.02  |
| `YOLOP(ours)` | 70.50   | 26.20  |

#### Ablation Studies 1: End-to-end v.s. Step-by-step

| Training_method | Recall(%) | AP(%) | mIoU(%) | Accuracy(%) | IoU(%) |
| --------------- | --------- | ----- | ------- | ----------- | ------ |
| `ES-W`          | 87.0      | 75.3  | 90.4    | 66.8        | 26.2   |
| `ED-W`          | 87.3      | 76.0  | 91.6    | 71.2        | 26.1   |
| `ES-D-W`        | 87.0      | 75.1  | 91.7    | 68.6        | 27.0   |
| `ED-S-W`        | 87.5      | 76.1  | 91.6    | 68.0        | 26.8   |
| `End-to-end`    | 89.2      | 76.5  | 91.5    | 70.5        | 26.2   |

#### Ablation Studies 2: Multi-task v.s. Single task

| Training_method | Recall(%) | AP(%) | mIoU(%) | Accuracy(%) | IoU(%) | Speed(ms/frame) |
| --------------- | --------- | ----- | ------- | ----------- | ------ | --------------- |
| `Det(only)`     | 88.2      | 76.9  | -       | -           | -      | 15.7            |
| `Da-Seg(only)`  | -         | -     | 92.0    | -           | -      | 14.8            |
| `Ll-Seg(only)`  | -         | -     | -       | 79.6        | 27.9   | 14.8            |
| `Multitask`     | 89.2      | 76.5  | 91.5    | 70.5        | 26.2   | 24.4            |

**Notes**:

- In table 4, E, D, S and W refer to Encoder, Detect head, two Segment heads and whole network. So the Algorithm (First, we only train Encoder and Detect head. Then we freeze the Encoder and Detect head as well as train two Segmentation heads. Finally, the entire network is trained jointly for all three tasks.) can be marked as ED-S-W, and the same for others.

### Visualization

#### Traffic Object Detection Result

<img width="800" alt="Traffic Object Detection Result" src="https://github.com/hustvl/YOLOP/raw/main/pictures/detect.png">
&nbsp;

#### Drivable Area Segmentation Result

<img width="800" alt="Drivable Area Segmentation Result" src="https://github.com/hustvl/YOLOP/raw/main/pictures/da.png">
&nbsp;

#### Lane Detection Result

<img width="800" alt="Lane Detection Result" src="https://github.com/hustvl/YOLOP/raw/main/pictures/ll.png">
&nbsp;

**Notes**:

- The visualization of lane detection result has been post processed by quadratic fitting.

### Deployment

Our model can reason in real-time on **Jetson Tx2**, with **Zed Camera** to capture image. We use **TensorRT** tool for speeding up. We provide code for deployment and reasoning of model in [github code](https://github.com/hustvl/YOLOP/tree/main/toolkits/deploy).


### Load From PyTorch Hub
This example loads the pretrained **YOLOP** model and passes an image for inference.
```python
import torch

# load model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

#inference
img = torch.randn(1,3,640,640)
det_out, da_seg_out,ll_seg_out = model(img)
```

### Citation

See for more detail in [github code](https://github.com/hustvl/YOLOP) and [arxiv paper](https://arxiv.org/abs/2108.11250).

If you find our paper and code useful for your research, please consider giving a star and citation:

