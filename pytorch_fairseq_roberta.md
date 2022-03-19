---
layout: hub_detail
background-class: hub-background
body-class: hub
title: RoBERTa
summary: A Robustly Optimized BERT Pretraining Approach
category: researchers
image: fairseq_logo.png
author: Facebook AI (fairseq Team)
tags: [nlp]
github-link: https://github.com/pytorch/fairseq/
github-id: pytorch/fairseq
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/RoBERTa
---


### Model Description

Bidirectional Encoder Representations from Transformers, or [BERT][1], is a
revolutionary self-supervised pretraining technique that learns to predict
intentionally hidden (masked) sections of text. Crucially, the representations
learned by BERT have been shown to generalize well to downstream tasks, and when
BERT was first released in 2018 it achieved state-of-the-art results on many NLP
benchmark datasets.

[RoBERTa][2] builds on BERT's language masking strategy and modifies key
hyperparameters in BERT, including removing BERT's next-sentence pretraining
objective, and training with much larger mini-batches and learning rates.
RoBERTa was also trained on an order of magnitude more data than BERT, for a
longer amount of time. This allows RoBERTa representations to generalize even
better to downstream tasks compared to BERT.


### Requirements

We require a few additional Python dependencies for preprocessing:

```bash
pip install regex requests hydra-core omegaconf
```


### Example

##### Load RoBERTa
```python
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply Byte-Pair Encoding (BPE) to input text
```python
tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
assert roberta.decode(tokens) == 'Hello world!'
```

##### Extract features from RoBERTa
```python
# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

##### Use RoBERTa for sentence-pair classification tasks
```python
# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

with torch.no_grad():
    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 0  # contradiction

    # Encode another pair of sentences
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 2  # entailment
```

##### Register a new (randomly initialized) classification head
```python
roberta.register_classification_head('new_task', num_classes=3)
logprobs = roberta.predict('new_task', tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
```


### References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding][1]
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach][2]


[1]: https://arxiv.org/abs/1810.04805
[2]: https://arxiv.org/abs/1907.11692
