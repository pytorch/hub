---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: Silero Language Classifier
summary: Pre-trained Spoken Language Classifier
image: silero_logo.jpg
author: Silero AI Team
tags: [audio, scriptable]
github-link: https://github.com/snakers4/silero-vad
github-id: snakers4/silero-vad
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
---


```bash
# this assumes that you have a proper version of PyTorch already installed
pip install -q torchaudio soundfile
```

```python
import torch
torch.set_num_threads(1)
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_lang_detector',
                              force_reload=True)

get_language, read_audio = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/de.wav')
language = get_language(wav, model)

pprint(language)
```

### Model Description

Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier. Enterprise-grade Speech Products made refreshingly simple (see our STT models). **Each model is published separately**.

Currently, there are hardly any high quality / modern / free / public voice activity detectors except for WebRTC Voice Activity Detector (link). WebRTC though starts to show its age and it suffers from many false positives.

Also in some cases it is crucial to be able to anonymize large-scale spoken corpora (i.e. remove personal data). Typically personal data is considered to be private / sensitive if it contains (i) a name (ii) some private ID. Name recognition is a highly subjective matter and it depends on locale and business case, but Voice Activity and Number Detection are quite general tasks.

**(!!!) Important Notice (!!!)** - the models are intended to run on CPU only and were optimized for performance on 1 CPU thread. Note that the model is quantized.


### Supported Languages

As of this page update, the following languages are supported:

- Russian
- English
- German
- Spanish

To see the always up-to-date language list, please visit our [repo](https://github.com/snakers4/silero-vad).

### Additional Examples and Benchmarks

For additional examples and other model formats please visit this [link](https://github.com/snakers4/silero-vad) and please refer to the extensive examples in the Colab format (including the streaming examples).

### References

Language classifier model architecture is based on similar STT architectures.

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Alexander Veysov, "Toward's an ImageNet Moment for Speech-to-Text", The Gradient, 2020](https://thegradient.pub/towards-an-imagenet-moment-for-speech-to-text/)
- [Alexander Veysov, "A Speech-To-Text Practitioner’s Criticisms of Industry and Academia", The Gradient, 2020](https://thegradient.pub/a-speech-to-text-practitioners-criticisms-of-industry-and-academia/)

