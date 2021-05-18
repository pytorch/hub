---
layout: hub_detail
background-class: hub-background
body-class: hub
title: PyTorch-Transformers
summary:  PyTorch implementations of popular NLP Transformers
category: researchers
image: huggingface-logo.png
author: HuggingFace Team
tags: [nlp]
github-link: https://github.com/huggingface/pytorch-transformers.git
github-id: huggingface/transformers
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

# Model Description


PyTorch-Transformers (formerly known as `pytorch-pretrained-bert`) is a library of state-of-the-art pre-trained models for Natural Language Processing (NLP).

The library currently contains PyTorch implementations, pre-trained model weights, usage scripts and conversion utilities for the following models:

1. **[BERT](https://github.com/google-research/bert)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. **[GPT](https://github.com/openai/finetune-transformer-lm)** (from OpenAI) released with the paper [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. **[GPT-2](https://blog.openai.com/better-language-models/)** (from OpenAI) released with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. **[Transformer-XL](https://github.com/kimiyoung/transformer-xl)** (from Google/CMU) released with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. **[XLNet](https://github.com/zihangdai/xlnet/)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. **[XLM](https://github.com/facebookresearch/XLM/)** (from Facebook) released together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
7. **[RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)** (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. **[DistilBERT](https://github.com/huggingface/pytorch-transformers/tree/master/examples/distillation)** (from HuggingFace), released together with the blogpost [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b) by Victor Sanh, Lysandre Debut and Thomas Wolf.

The components available here are based on the `AutoModel` and `AutoTokenizer` classes of the `pytorch-transformers` library.

# Requirements

Unlike most other PyTorch Hub models, BERT requires a few additional Python packages to be installed.

```bash
pip install tqdm boto3 requests regex sentencepiece sacremoses
```

# Usage

The available methods are the following:
- `config`: returns a configuration item corresponding to the specified model or pth.
- `tokenizer`: returns a tokenizer corresponding to the specified model or path
- `model`: returns a model corresponding to the specified model or path
- `modelForCausalLM`: returns a model with a language modeling head corresponding to the specified model or path
- `modelForSequenceClassification`: returns a model with a sequence classifier corresponding to the specified model or path
- `modelForQuestionAnswering`: returns a model with  a question answering head corresponding to the specified model or path

All these methods share the following argument: `pretrained_model_or_path`, which is a string identifying a pre-trained model or path from which an instance will be returned. There are several checkpoints available for each model, which are detailed below:




The available models are listed on the [pytorch-transformers documentation, pre-trained models section](https://huggingface.co/pytorch-transformers/pretrained_models.html).

# Documentation

Here are a few examples detailing the usage of each available method.


## Tokenizer

The tokenizer object allows the conversion from character strings to tokens understood by the different models. Each model has its own tokenizer, and some tokenizing methods are different across tokenizers. The complete documentation can be found [here](https://huggingface.co/pytorch-transformers/main_classes/tokenizer.html).

```py
import torch
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`
```

## Models

The model object is a model instance inheriting from a `nn.Module`. Each model is accompanied by their saving/loading methods, either from a local file or directory, or from a pre-trained configuration (see previously described `config`). Each model works differently, a complete overview of the different models can be found in the [documentation](https://huggingface.co/pytorch-transformers/pretrained_models.html).

```py
import torch
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
model = torch.hub.load('huggingface/pytorch-transformers', 'model', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', output_attentions=True)  # Update configuration during loading
assert model.config.output_attentions == True
# Loading from a TF checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
model = torch.hub.load('huggingface/pytorch-transformers', 'model', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## Models with a language modeling head

Previously mentioned `model` instance with an additional language modeling head.

```py
import torch
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')    # Download model and configuration from huggingface.co and cache.
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2', output_attentions=True)  # Update configuration during loading
assert model.config.output_attentions == True
# Loading from a TF checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained('./tf_model/gpt_tf_model_config.json')
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './tf_model/gpt_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## Models with a sequence classification head

Previously mentioned `model` instance  with an additional sequence classification head.

```py
import torch
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased', output_attention=True)  # Update configuration during loading
assert model.config.output_attention == True
# Loading from a TF checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## Models with a question answering head

Previously mentioned `model` instance  with an additional question answering head.

```py
import torch
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-uncased', output_attention=True)  # Update configuration during loading
assert model.config.output_attention == True
# Loading from a TF checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## Configuration

The configuration is optional. The configuration object holds information concerning the model, such as the number of heads/layers, if the model should output attentions or hidden states, or if it should be adapted for TorchScript. Many parameters are available, some specific to each model. The complete documentation can be found [here](https://huggingface.co/pytorch-transformers/main_classes/configuration.html).

```py
import torch
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')  # Download configuration from S3 and cache.
config = torch.hub.load('huggingface/pytorch-transformers', 'config', './test/bert_saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
config = torch.hub.load('huggingface/pytorch-transformers', 'config', './test/bert_saved_model/my_configuration.json')
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased', output_attention=True, foo=False)
assert config.output_attention == True
config, unused_kwargs = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased', output_attention=True, foo=False, return_unused_kwargs=True)
assert config.output_attention == True
assert unused_kwargs == {'foo': False}

# Using the configuration with a model
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')
config.output_attentions = True
config.output_hidden_states = True
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', config=config)
# Model will now output attentions and hidden states as well

```

# Example Usage

Here is an example on how to tokenize the input text to be fed as input to a BERT model, and then get the hidden states computed by such a model or predict masked tokens using language modeling BERT model.

## First, tokenize the input

```python
import torch
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
```

## Using `BertModel` to encode the input sentence in a sequence of last layer hidden-states

```python
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)
```

## Using `modelForMaskedLM` to predict a masked token with BERT

```python
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
indexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([indexed_tokens])

masked_lm_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')

with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

# Get the predicted token
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'Jim'
```

## Using `modelForQuestionAnswering` to do question answering with BERT

```python
question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

# The format is paragraph first and then question
text_1 = "Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ?"
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

# Predict the start and end positions logits
with torch.no_grad():
    out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

# get the highest prediction
answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
assert answer == "puppeteer"

# Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions (set model to train mode before if used for training)
start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
multiple_choice_loss = question_answering_model(tokens_tensor, token_type_ids=segments_tensors, start_positions=start_positions, end_positions=end_positions)
```

## Using `modelForSequenceClassification` to do paraphrase classification with BERT

```python
sequence_classification_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-cased-finetuned-mrpc')
sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased-finetuned-mrpc')

text_1 = "Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ?"
indexed_tokens = sequence_classification_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

# Predict the sequence classification logits
with torch.no_grad():
    seq_classif_logits = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors)

predicted_labels = torch.argmax(seq_classif_logits[0]).item()

assert predicted_labels == 0  # In MRPC dataset this means the two sentences are not paraphrasing each other

# Or get the sequence classification loss (set model to train mode before if used for training)
labels = torch.tensor([1])
seq_classif_loss = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors, labels=labels)
```
