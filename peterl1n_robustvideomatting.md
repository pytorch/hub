---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: Robust Video Matting
summary: RVM is a matting network for background replacement tasks. It is trained on human subjects and supports high resolutions such as 4K/HD.
image: rvm_logo.gif
author: Peter Lin
tags: [vision, scriptable]
github-link: https://github.com/PeterL1n/RobustVideoMatting
github-id: PeterL1n/RobustVideoMatting
featured_image_1: rvm_feature.gif
featured_image_2: no-image
accelerator: cuda-optional
---

### Model Description

Robust Video Matting (RVM) is human matting network proposed in paper ["Robust High-Resolution Video Matting with Temporal Guidance"](https://peterl1n.github.io/RobustVideoMatting/). It can process high resolutions such as 4K/HD in real-time (depending on the GPU). It is also natively designed for processing videos by using a recurrent architecture to exploit temporal information.

### Demo

Watch the showreel video on [YouTube](https://youtu.be/Jvzltozpbpk) or [Bilibili](https://www.bilibili.com/video/BV1Z3411B7g7/) to see how it performs!


### Load the Model

Make sure you have `torch>=1.8.1` and `torchvision>=0.9.1` installed.

```python
import torch

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda().eval()
```

We recommend using the `mobilenetv3` backbone for most cases. We also provide a `resnet50` backbone with slight improvement in performance.

### Use Inference API

We provide a simple inference function `convert_video`. You can use it to apply our model on your videos.

```sh
pip install av pims tqdm
```

```python
convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
```

```python
convert_video(
    model,                           # The loaded model, can be on any device (cpu or cuda).
    input_source='input.mp4',        # A video file or an image sequence directory.
    input_resize=None,               # [Optional] Resize the input to (width, height).
    downsample_ratio=None,           # [Optional] Advanced hyperparameter. See inference doc.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mp4',    # File path if video; directory path if png sequence.
    output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    seq_chunk=4,                     # Process n frames at once for better parallelism.
    num_workers=0,                   # Only for image sequence input.
    progress=True                    # Print conversion progress.
)
```

More advanced usage, see [inference documentation](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md). Note that the inference function will likely not be real-time for HD/4K because video encoding/decoding are not done with hardware acceleration. You can implement your own inference loop as shown below.

### Implement Custom Inference Loops

```python
bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 0.25                                # Adjust based on your video.

with torch.no_grad():
    for src in __YOUR_VIDEO_READER__:                  # RGB video frame normalized to 0 ~ 1.
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
        out = fgr * pha + bgr * (1 - pha)              # Composite to green background. 
        __YOUR_VIDEO_WRITER__(out)
```

* Since RVM uses a recurrent architecture, we initialize the recurrent states `rec` to `None` initially, and cycle the recurrent states when processing every frame.
* `src` can be `BCHW`, or `BTCHW` where `T` is a chunk of frames that can be given to the model at once to improve parallalism.
* `downsample_ratio` is an important hyperparameter. See [inference documentation](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md) for more detail.

### References

```
@misc{rvm,
      title={Robust High-Resolution Video Matting with Temporal Guidance}, 
      author={Shanchuan Lin and Linjie Yang and Imran Saleemi and Soumyadip Sengupta},
      year={2021},
      eprint={2108.11515},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```