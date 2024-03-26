---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Transformer (NMT)
summary: Transformer models for English-French and English-German translation.
category: researchers
image: fairseq_logo.png
author: Facebook AI (fairseq Team)
tags: [nlp]
github-link: https://github.com/pytorch/fairseq/
github-id: pytorch/fairseq
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
order: 2
demo-model-link: https://huggingface.co/spaces/pytorch/Transformer_NMT
---


### Model Description

The Transformer, introduced in the paper [Attention Is All You Need][1], is a
powerful sequence-to-sequence modeling architecture capable of producing
state-of-the-art neural machine translation (NMT) systems.

Recently, the fairseq team has explored large-scale semi-supervised training of
Transformers using back-translated data, further improving translation quality
over the original model. More details can be found in [this blog post][2].


### Requirements

We require a few additional Python dependencies for preprocessing:

```bash
pip install bitarray fastBPE hydra-core omegaconf regex requests sacremoses subword_nmt
```


### English-to-French Translation

To translate from English to French using the model from the paper [Scaling
Neural Machine Translation][3]:

```python
import torch

# Load an En-Fr Transformer model trained on WMT'14 data :
en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

# Use the GPU (optional):
en2fr.cuda()

# Translate with beam search:
fr = en2fr.translate('Hello world!', beam=5)
assert fr == 'Bonjour à tous !'

# Manually tokenize:
en_toks = en2fr.tokenize('Hello world!')
assert en_toks == 'Hello world !'

# Manually apply BPE:
en_bpe = en2fr.apply_bpe(en_toks)
assert en_bpe == 'H@@ ello world !'

# Manually binarize:
en_bin = en2fr.binarize(en_bpe)
assert en_bin.tolist() == [329, 14044, 682, 812, 2]

# Generate five translations with top-k sampling:
fr_bin = en2fr.generate(en_bin, beam=5, sampling=True, sampling_topk=20)
assert len(fr_bin) == 5

# Convert one of the samples to a string and detokenize
fr_sample = fr_bin[0]['tokens']
fr_bpe = en2fr.string(fr_sample)
fr_toks = en2fr.remove_bpe(fr_bpe)
fr = en2fr.detokenize(fr_toks)
assert fr == en2fr.decode(fr_sample)
```


### English-to-German Translation

Semi-supervised training with back-translation is an effective way of improving
translation systems. In the paper [Understanding Back-Translation at Scale][4],
we back-translate over 200 million German sentences to use as additional
training data. An ensemble of five of these models was the winning submission to
the [WMT'18 English-German news translation competition][5].

We can further improved this approach through [noisy-channel reranking][6]. More
details can be found in [this blog post][7]. An ensemble of models trained with
this technique was the winning submission to the [WMT'19 English-German news
translation competition][8].

To translate from English to German using one of the models from the winning submission:

```python
import torch

# Load an En-De Transformer model trained on WMT'19 data:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')

# Access the underlying TransformerModel
assert isinstance(en2de.models[0], torch.nn.Module)

# Translate from En-De
de = en2de.translate('PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility.')
assert de == 'PyTorch Hub ist ein vorgefertigtes Modell-Repository, das die Reproduzierbarkeit der Forschung erleichtern soll.'
```

We can also do a round-trip translation to create a paraphrase:
```python
# Round-trip translations between English and German:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

paraphrase = de2en.translate(en2de.translate('PyTorch Hub is an awesome interface!'))
assert paraphrase == 'PyTorch Hub is a fantastic interface!'

# Compare the results with English-Russian round-trip translation:
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

paraphrase = ru2en.translate(en2ru.translate('PyTorch Hub is an awesome interface!'))
assert paraphrase == 'PyTorch is a great interface!'
```


### References

- [Attention Is All You Need][1]
- [Scaling Neural Machine Translation][3]
- [Understanding Back-Translation at Scale][4]
- [Facebook FAIR's WMT19 News Translation Task Submission][6]


[1]: https://arxiv.org/abs/1706.03762
[2]: https://code.fb.com/ai-research/scaling-neural-machine-translation-to-bigger-data-sets-with-faster-training-and-inference/
[3]: https://arxiv.org/abs/1806.00187
[4]: https://arxiv.org/abs/1808.09381
[5]: http://www.statmt.org/wmt18/translation-task.html
[6]: https://arxiv.org/abs/1907.06616
[7]: https://ai.facebook.com/blog/facebook-leads-wmt-translation-competition/
[8]: http://www.statmt.org/wmt19/translation-task.html
