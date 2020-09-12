---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: Silero Speech-To-Text Models
summary: A set of compact enterprise-grade pre-trained STT Models for multiple languages. Languages available - English, German and Spanish.
image: silero_logo.jpg
author: Silero AI Team
tags: [audio, scriptable]
github-link: https://github.com/snakers4/silero-models
github-id: snakers4/silero-models
featured_image_1: silero_stt_model.jpg
featured_image_2: silero_imagenet_moment.png
accelerator: cuda-optional
---

```python
import torch
from glob import glob
from utils import (split_into_batches,
                   read_batch,
                   prepare_model_input)
# see https://github.com/snakers4/silero-models for utils and more examples

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder = torch.hub.load(github='snakers4/silero-models',
                                model='silero_stt',
                                device=device)

test_files = glob('path/to/your/file/*.opus')  # or any format compatible with TorchAudio
batches = split_into_batches(test_files, batch_size=10)

# batch is just a list of paths
# prepare_model_input just reads audio
# resamples and normalizes and puts it into a batch
# model operates at 16 kHz sample rate
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    print(decoder(example.cpu()))
```

### Model Description

Silero Speech-To-Text models provide enterprise grade STT in a compact form-factor for several commonly spoken languages. Unlike conventional ASR models our models are robust to a variety of dialects, codecs, domains, noises, lower sampling rates (for simplicity audio should be resampled to 16 kHz). The models consume a normalized audio in the form of samples (i.e. without any pre-processing except for normalization) and output frames with token probabilities. We provide a decoder utility for simplicity (we could include it into our model itself, but scripted modules had problems with storing model artifacts i.e. labels during certain export scenarios).

We hope that our efforts with Open-STT and Silero Models will bring the ImageNet moment in speech closer.

### Supported Languages and Formats

As of this page update, the following languages are supported:

- English
- German
- Spanish

To see the always up-to-date language list, please visit our [repo](https://github.com/snakers4/silero-models) and see the `yml` [file](https://github.com/snakers4/silero-models/blob/master/models.yml) for all available checkpoints.

### Additional Examples and Benchmarks

For additional examples and other model formats please visit this [link](https://github.com/snakers4/silero-models). For quality and performance benchmarks please see the [wiki](https://github.com/snakers4/silero-models/wiki). These resources will be updated from time to time.

### References

- [Silero Models](https://github.com/snakers4/silero-models)
- [Alexander Veysov, "Toward's an ImageNet Moment for Speech-to-Text", The Gradient, 2020](https://thegradient.pub/towards-an-imagenet-moment-for-speech-to-text/)
- [Alexander Veysov, "A Speech-To-Text Practitioner’s Criticisms of Industry and Academia", The Gradient, 2020](https://thegradient.pub/a-speech-to-text-practitioners-criticisms-of-industry-and-academia/)

