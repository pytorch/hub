---
layout: pytorch_hub_detail
background-class: pytorch-hub-background
body-class: pytorch-hub
title: GPT
summary: GPT models
category: research
image: huggingface-logo.png
author: HuggingFace Team
tags: [NLP, GPT]
github-link: https://github.com/huggingface/pytorch-pretrained-BERT.git
featured_image_1: no-image
featured_image_2: no-image
---

### Model Description

GPT was released together with the paper [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) by Alec Radford et al at OpenAI. It's a combination of two ideas: Transformer model and large scale unsupervised pre-training.

Here are three models based on [OpenAI's pre-trained weights](https://github.com/openai/finetune-transformer-lm) along with the associated Tokenizer.
It includes:
- `openAIGPTModel`: raw OpenAI GPT Transformer model (fully pre-trained)
- `openAIGPTLMHeadModel`: OpenAI GPT Transformer with the tied language modeling head on top (fully pre-trained)
- `openAIGPTDoubleHeadsModel`: OpenAI GPT Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT Transformer is pre-trained, the multiple choice classification head is only initialized and has to be trained)

### Example:

Here are three examples on how to use `openAIGPTTokenizer`, `openAIGPTModel` and `openAIGPTLMHeadModel`.

First, we prepare the inputs by tokenizing the text.

```python
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')

text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
```

We can get the hidden states computed by `openAIGPTModel`.

```python
# Load the tokenizer
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')

#  Prepare tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# Load openAIGPTModel
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTModel', 'openai-gpt')
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
	hidden_states = model(tokens_tensor)
```

We can predict the next tokens using `openAIGPTLMHeadModel`.

```python
# Load the tokenizer
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')

#  Prepare tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# Load openAIGPTLMHeadModel
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTLMHeadModel', 'openai-gpt')
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
	predictions = model(tokens_tensor)

# Get the predicted last token
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
```

### Resources:

 - Paper: [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
 - [Blogpost from OpenAI](https://openai.com/blog/language-unsupervised/)
 - Initial repository (with detailed examples and documentation): [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
