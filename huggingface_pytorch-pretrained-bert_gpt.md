---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GPT
summary: Generative Pre-Training (GPT) models for language understanding
category: researchers
image: huggingface-logo.png
author: HuggingFace Team
tags: [nlp]
github-link: https://github.com/huggingface/pytorch-pretrained-BERT.git
featured_image_1: GPT1.png
featured_image_2: no-image
---

### Model Description

GPT was released together with the paper [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) by Alec Radford et al at OpenAI. It's a combination of two ideas: Transformer model and large scale unsupervised pre-training.

Here are three models based on [OpenAI's pre-trained weights](https://github.com/openai/finetune-transformer-lm) along with the associated Tokenizer.
It includes:
- `openAIGPTModel`: raw OpenAI GPT Transformer model (fully pre-trained)
- `openAIGPTLMHeadModel`: OpenAI GPT Transformer with the tied language modeling head on top (fully pre-trained)
- `openAIGPTDoubleHeadsModel`: OpenAI GPT Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT Transformer is pre-trained, the multiple choice classification head is only initialized and has to be trained)

### Requirements

Unlike most other PyTorch Hub models, BERT requires a few additional Python packages to be installed.

```bash
pip install tqdm boto3 requests regex
```

### Example

Here is an example on how to tokenize the text with `openAIGPTTokenizer`, and then get the hidden states computed by `openAIGPTModel` or predict the next token using `openAIGPTLMHeadModel`.

```python
### First, tokenize the input
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')

#  Prepare tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

### Get the hidden states computed by `openAIGPTModel`
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTModel', 'openai-gpt')
model.eval()

# Compute hidden states features for each layer
with torch.no_grad():
	hidden_states = model(tokens_tensor)

### Predict the next token using `openAIGPTLMHeadModel`
lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTLMHeadModel', 'openai-gpt')
lm_model.eval()

# Predict all tokens
with torch.no_grad():
	predictions = lm_model(tokens_tensor)

# Get the last predicted token
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == '.</w>'
```

### Requirement
The model only support python3.

### Resources

 - Paper: [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
 - [Blogpost from OpenAI](https://openai.com/blog/language-unsupervised/)
 - Initial repository (with detailed examples and documentation): [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
