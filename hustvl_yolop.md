---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: YOLOP
summary: YOLOP pretrained on the BDD100K dataset
image: yolop.png
author: hustvl
tags: < [Object detection, Drivable area segmentation, Lane detection, Multitask learning] >
github-link: https://github.com/hustvl/YOLOP
github-id: hustvl/YOLOP
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda, tensorRT
---

### You Only 👀 Once for Panoptic 🚗 Perception

> [**You Only Look at Once for Panoptic  driving Perception**](https://arxiv.org/abs/2108.11250)
>
> by Dong Wu, Manwen Liao, Weitian Zhang, [Xinggang Wang](https://xinggangw.info/) 📧 [*School of EIC, HUST*](http://eic.hust.edu.cn/English/Home.htm)
>
> (📧) corresponding author.
>
> *arXiv technical report ([arXiv 2108.11250](https://arxiv.org/abs/2108.11250))*
>
> github code([arXiv 2108.11250](https://arxiv.org/abs/2108.11250))

------

#### The Illustration of YOLOP

[![yolop](https://github.com/hustvl/YOLOP/raw/main/pictures/yolop.png)](https://github.com/hustvl/YOLOP/blob/main/pictures/yolop.png)

#### Contributions

- We put forward an efficient multi-task network that can jointly handle three crucial tasks in autonomous driving: object detection, drivable area segmentation and lane detection to save computational costs, reduce inference time as well as improve the performance of each task. Our work is the first to reach real-time on embedded devices while maintaining state-of-the-art level performance on the `BDD100K `dataset.
- We design the ablative experiments to verify the effectiveness of our multi-tasking scheme. It is proved that the three tasks can be learned jointly without tedious alternating optimization.

#### Results

##### Traffic Object Detection Result

| Model          | Recall(%) | mAP50(%) | Speed(fps) |
| -------------- | --------- | -------- | ---------- |
| `Multinet`     | 81.3      | 60.2     | 8.6        |
| `DLT-Net`      | 89.4      | 68.4     | 9.3        |
| `Faster R-CNN` | 77.2      | 55.6     | 5.3        |
| `YOLOv5s`      | 86.8      | 77.2     | 82         |
| `YOLOP(ours)`  | 89.2      | 76.5     | 41         |

##### Drivable Area Segmentation Result

| Model         | mIOU(%) | Speed(fps) |
| ------------- | ------- | ---------- |
| `Multinet`    | 71.6    | 8.6        |
| `DLT-Net`     | 71.3    | 9.3        |
| `PSPNet`      | 89.6    | 11.1       |
| `YOLOP(ours)` | 91.5    | 41         |

##### Lane Detection Result:

| Model         | mIOU(%) | IOU(%) |
| ------------- | ------- | ------ |
| `ENet`        | 34.12   | 14.64  |
| `SCNN`        | 35.79   | 15.84  |
| `ENet-SAD`    | 36.56   | 16.02  |
| `YOLOP(ours)` | 70.50   | 26.20  |

##### Ablation Studies 1: End-to-end v.s. Step-by-step:

| Training_method | Recall(%) | AP(%) | mIoU(%) | Accuracy(%) | IoU(%) |
| --------------- | --------- | ----- | ------- | ----------- | ------ |
| `ES-W`          | 87.0      | 75.3  | 90.4    | 66.8        | 26.2   |
| `ED-W`          | 87.3      | 76.0  | 91.6    | 71.2        | 26.1   |
| `ES-D-W`        | 87.0      | 75.1  | 91.7    | 68.6        | 27.0   |
| `ED-S-W`        | 87.5      | 76.1  | 91.6    | 68.0        | 26.8   |
| `End-to-end`    | 89.2      | 76.5  | 91.5    | 70.5        | 26.2   |

##### Ablation Studies 2: Multi-task v.s. Single task:

| Training_method | Recall(%) | AP(%) | mIoU(%) | Accuracy(%) | IoU(%) | Speed(ms/frame) |
| --------------- | --------- | ----- | ------- | ----------- | ------ | --------------- |
| `Det(only)`     | 88.2      | 76.9  | -       | -           | -      | 15.7            |
| `Da-Seg(only)`  | -         | -     | 92.0    | -           | -      | 14.8            |
| `Ll-Seg(only)`  | -         | -     | -       | 79.6        | 27.9   | 14.8            |
| `Multitask`     | 89.2      | 76.5  | 91.5    | 70.5        | 26.2   | 24.4            |

**Notes**:

- In table 4, E, D, S and W refer to Encoder, Detect head, two Segment heads and whole network. So the Algorithm (First, we only train Encoder and Detect head. Then we freeze the Encoder and Detect head as well as train two Segmentation heads. Finally, the entire network is trained jointly for all three tasks.) can be marked as ED-S-W, and the same for others.

------

#### Visualization

##### Traffic Object Detection Result

[![detect result](https://github.com/hustvl/YOLOP/raw/main/pictures/detect.png)](https://github.com/hustvl/YOLOP/blob/main/pictures/detect.png)

##### Drivable Area Segmentation Result

[![img](https://github.com/hustvl/YOLOP/raw/main/pictures/da.png)](https://github.com/hustvl/YOLOP/blob/main/pictures/da.png)

##### Lane Detection Result

[![img](https://github.com/hustvl/YOLOP/raw/main/pictures/ll.png)](https://github.com/hustvl/YOLOP/blob/main/pictures/ll.png)

**Notes**:

- The visualization of lane detection result has been post processed by quadratic fitting.

------

#### Demonstration

| input                                                        | output                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![img](https://github.com/hustvl/YOLOP/raw/main/pictures/input1.gif/)](https://github.com/hustvl/YOLOP/blob/main/pictures/input1.gif/) | [![img](https://github.com/hustvl/YOLOP/raw/main/pictures/output1.gif/)](https://github.com/hustvl/YOLOP/blob/main/pictures/output1.gif/) |
| [![img](https://github.com/hustvl/YOLOP/raw/main/pictures/input2.gif/)](https://github.com/hustvl/YOLOP/blob/main/pictures/input2.gif/) | [![img](https://github.com/hustvl/YOLOP/raw/main/pictures/output2.gif/)](https://github.com/hustvl/YOLOP/blob/main/pictures/output2.gif/) |

#### Deployment

Our model can reason in real-time on `Jetson Tx2`, with `Zed Camera` to capture image. We use `TensorRT` tool for speeding up. We provide code for deployment and reasoning of model in [github code](https://github.com/hustvl/YOLOP/tree/main/toolkits/deploy).

#### Citation

See for more detail in [github code](https://github.com/hustvl/YOLOP) and [arxiv paper](https://arxiv.org/abs/2108.11250).

If you find our paper and code useful for your research, please consider giving a star ⭐ and citation 📝 :

```
@misc{2108.11250,
Author = {Dong Wu and Manwen Liao and Weitian Zhang and Xinggang Wang},
Title = {YOLOP: You Only Look Once for Panoptic Driving Perception},
Year = {2021},
Eprint = {arXiv:2108.11250},
}
```



### References

The works we has use for reference including `Multinet` ([paper](https://arxiv.org/pdf/1612.07695.pdf?utm_campaign=affiliate-ir-Optimise media( South East Asia) Pte. ltd._156_-99_national_R_all_ACQ_cpa_en&utm_content=&utm_source= 388939),[code](https://github.com/MarvinTeichmann/MultiNet)）,`DLT-Net` ([paper](https://ieeexplore.ieee.org/abstract/document/8937825)）,`Faster R-CNN` ([paper](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf),[code](https://github.com/ShaoqingRen/faster_rcnn)）,`YOLOv5s`（[code](https://github.com/ultralytics/yolov5)) ,`PSPNet`([paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf),[code](https://github.com/hszhao/PSPNet)) ,`ENet`([paper](https://arxiv.org/pdf/1606.02147.pdf),[code](https://github.com/osmr/imgclsmob)) `SCNN`([paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16802/16322),[code](https://github.com/XingangPan/SCNN)) `SAD-ENet`([paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hou_Learning_Lightweight_Lane_Detection_CNNs_by_Self_Attention_Distillation_ICCV_2019_paper.pdf),[code](https://github.com/cardwing/Codes-for-Lane-Detection)). Thanks for their wonderful works.