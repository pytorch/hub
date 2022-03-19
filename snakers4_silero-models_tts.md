---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: Silero Text-To-Speech Models
summary: A set of compact enterprise-grade pre-trained TTS Models for multiple languages
image: silero_logo.jpg
author: Silero AI Team
tags: [audio, scriptable]
github-link: https://github.com/snakers4/silero-models
github-id: snakers4/silero-models
featured_image_1: no-image
featured_image_1: no-image
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/silero_tts
---

```bash
# this assumes that you have a proper version of PyTorch already installed
pip install -q torchaudio omegaconf
```

```python
import torch

language = 'en'
speaker = 'lj_16khz'
device = torch.device('cpu')
model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_tts',
                                                                      language=language,
                                                                      speaker=speaker)
model = model.to(device)  # gpu or cpu
audio = apply_tts(texts=[example_text],
                  model=model,
                  sample_rate=sample_rate,
                  symbols=symbols,
                  device=device)
```

### Model Description

Silero Text-To-Speech models provide enterprise grade TTS in a compact form-factor for several commonly spoken languages:

- One-line usage
- Naturally sounding speech
- No GPU or training required
- Minimalism and lack of dependencies
- A library of voices in many languages
- Support for `16kHz` and `8kHz` out of the box
- High throughput on slow hardware. Decent performance on one CPU thread

### Supported Languages and Formats

As of this page update, the speakers of the following languages are supported both in 8 kHz and 16 kHz:

- Russian (6 speakers)
- English (1 speaker)
- German (1 speaker)
- Spanish (1 speaker)
- French (1 speaker)

To see the always up-to-date language list, please visit our [repo](https://github.com/snakers4/silero-models) and see the `yml` [file](https://github.com/snakers4/silero-models/blob/master/models.yml) for all available checkpoints.

### Additional Examples and Benchmarks

For additional examples and other model formats please visit this [link](https://github.com/snakers4/silero-models). For quality and performance benchmarks please see the [wiki](https://github.com/snakers4/silero-models/wiki). These resources will be updated from time to time.

### References

- [Silero Models](https://github.com/snakers4/silero-models)
- [High-Quality Speech-to-Text Made Accessible, Simple and Fast](https://habr.com/ru/post/549482/)