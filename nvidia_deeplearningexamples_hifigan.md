---
layout: hub_detail
background-class: hub-background
body-class: hub
title: HiFi GAN
summary: The HiFi GAN model for generating waveforms from mel spectrograms
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [audio]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
github-id: NVIDIA/DeepLearningExamples
featured_image_1: hifigan_model.png
featured_image_2: no-image
accelerator: cuda
---


### Model Description
This notebook demonstrates a PyTorch implementation of the HiFi-GAN model described in the paper: [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646).
The HiFi-GAN model implements a spectrogram inversion model that allows to synthesize speech waveforms from mel-spectrograms. It follows the generative adversarial network (GAN) paradigm, and is composed of a generator and a discriminator. After training, the generator is used for synthesis, and the discriminator is discarded.

Our implementation is based on the one [published by the authors of the paper](https://github.com/jik876/hifi-gan). We modify the original hyperparameters and provide an alternative training recipe, which enables training on larger batches and faster convergence. HiFi-GAN is trained on a publicly available [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/). The <a href="audio/">samples</a> demonstrate speech synthesized with our publicly available FastPitch and HiFi-GAN checkpoints.

#### Model architecture

![HiFiGAN Architecture](https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/SpeechSynthesis/HiFiGAN/img/hifigan_model.png)

### Example
In the example below:

- pretrained FastPitch and HiFiGAN models are loaded from torch.hub
- given tensor representation of an input text ("Say this smoothly to prove you are not a robot."), FastPitch generates mel spectrogram 
- HiFiGAN generates sound given the mel spectrogram
- the output sound is saved in an 'audio.wav' file

To run the example you need some extra python packages installed. These are needed for preprocessing of text and audio, as well as for display and input/output handling. Finally, for better performance of FastPitch model, we download the CMU pronounciation dictionary.
```bash
pip install numpy scipy librosa unidecode inflect librosa matplotlib==3.6.3
apt-get update
apt-get install -y libsndfile1 wget
wget https://raw.githubusercontent.com/NVIDIA/NeMo/263a30be71e859cee330e5925332009da3e5efbc/scripts/tts_dataset_files/heteronyms-052722 -qO heteronyms
wget https://raw.githubusercontent.com/NVIDIA/NeMo/263a30be71e859cee330e5925332009da3e5efbc/scripts/tts_dataset_files/cmudict-0.7b_nv22.08 -qO cmudict-0.7b
```

```python
import torch
import matplotlib.pyplot as plt
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
```

Download and setup FastPitch generator model. 
```python
fastpitch, generator_train_setup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch')
```

Download and setup vocoder and denoiser models.
```python
hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
```

Verify that generator and vocoder models agree on input parameters.
```python
CHECKPOINT_SPECIFIC_ARGS = [
    'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
    'symbol_set', 'max_wav_value', 'prepend_space_to_text',
    'append_space_to_text']

for k in CHECKPOINT_SPECIFIC_ARGS:

    v1 = generator_train_setup.get(k, None)
    v2 = vocoder_train_setup.get(k, None)

    assert v1 is None or v2 is None or v1 == v2, \
        f'{k} mismatch in spectrogram generator and vocoder'
```

Put all models on available device.
```python
fastpitch.to(device)
hifigan.to(device)
denoiser.to(device)
```

Load text processor.
```python
tp = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_textprocessing_utils', cmudict_path="cmudict-0.7b", heteronyms_path="heteronyms")
```

Set the text to be synthetized, prepare input and set additional generation parameters.
```python
text = "Say this smoothly, to prove you are not a robot."
```

```python
batches = tp.prepare_input_sequence([text], batch_size=1)
```

```python
gen_kw = {'pace': 1.0,
          'speaker': 0,
          'pitch_tgt': None,
          'pitch_transform': None}
denoising_strength = 0.005
```

```python
for batch in batches:
    with torch.no_grad():
        mel, mel_lens, *_ = fastpitch(batch['text'].to(device), **gen_kw)
        audios = hifigan(mel).float()
        audios = denoiser(audios.squeeze(1), denoising_strength)
        audios = audios.squeeze(1) * vocoder_train_setup['max_wav_value']

```

Plot the intermediate spectorgram.
```python
plt.figure(figsize=(10,12))
res_mel = mel[0].detach().cpu().numpy()
plt.imshow(res_mel, origin='lower')
plt.xlabel('time')
plt.ylabel('frequency')
_=plt.title('Spectrogram')
```

Syntesize audio.
```python
audio_numpy = audios[0].cpu().numpy()
Audio(audio_numpy, rate=22050)
```

Write audio to wav file.
```python
from scipy.io.wavfile import write
write("audio.wav", vocoder_train_setup['sampling_rate'], audio_numpy)
```

### Details
For detailed information on model input and output, training recipies, inference and performance visit: [github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/HiFiGAN) and/or [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/hifigan_pyt)

### References

 - [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
 - [Original implementation](https://github.com/jik876/hifi-gan)
 - [FastPitch on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/fastpitch_pyt)
 - [HiFi-GAN on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/hifigan_pyt)
 - [FastPitch and HiFi-GAN on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/HiFi-GAN)