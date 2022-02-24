---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: Silero Voice Activity Detector
summary: Pre-trained Voice Activity Detector
image: silero_logo.jpg
author: Silero AI Team
tags: [audio, scriptable]
github-link: https://github.com/snakers4/silero-vad
github-id: snakers4/silero-vad
featured_image_1: silero_vad_performance.png
featured_image_2: no-image
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/silero_vad
---


```bash
# this assumes that you have a proper version of PyTorch already installed
pip install -q torchaudio
```

```python
import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint
# download example
torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 _, read_audio,
 *_) = utils

sampling_rate = 16000 # also accepts 8000
wav = read_audio('en_example.wav', sampling_rate=sampling_rate)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
pprint(speech_timestamps)
```

### Model Description

Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD). Enterprise-grade Speech Products made refreshingly simple (see our STT models). **Each model is published separately**.

Currently, there are hardly any high quality / modern / free / public voice activity detectors except for WebRTC Voice Activity Detector (link). WebRTC though starts to show its age and it suffers from many false positives.

**(!!!) Important Notice (!!!)** - the models are intended to run on CPU only and were optimized for performance on 1 CPU thread. Note that the model is quantized.


### Additional Examples and Benchmarks

For additional examples and other model formats please visit this [link](https://github.com/snakers4/silero-vad) and please refer to the extensive examples in the Colab format (including the streaming examples).

### References

VAD model architectures are based on similar STT architectures.

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Alexander Veysov, "Toward's an ImageNet Moment for Speech-to-Text", The Gradient, 2020](https://thegradient.pub/towards-an-imagenet-moment-for-speech-to-text/)
- [Alexander Veysov, "A Speech-To-Text Practitioner’s Criticisms of Industry and Academia", The Gradient, 2020](https://thegradient.pub/a-speech-to-text-practitioners-criticisms-of-industry-and-academia/)

