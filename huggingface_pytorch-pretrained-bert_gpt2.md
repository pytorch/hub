---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GPT-2
summary:  Language Models are Unsupervised Multitask Learners
category: researchers
image: huggingface-logo.png
author: HuggingFace Team
tags: [nlp]
github-link: https://github.com/huggingface/pytorch-pretrained-BERT.git
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

### Model Description

GPT-2 was released together with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford by Alec Radford et al at OpenAI. It is a development of [GPT](https://github.com/pytorch/hub/blob/master/huggingface_pytorch-pretrained-bert_gpt.md) introduced in [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). It further asseses the impressive natural language generation abilities of large language models along with the ability to perform reasonably well on a diverse range of tasks in a zero-shot setting.

Here are three models based on OpenAI's pre-trained weights along with the associated Tokenizer.
It includes:
- `gpt2Model`: raw OpenAI GPT-2 Transformer model (fully pre-trained)
- `gpt2LMHeadModel`: OpenAI GPT-2 Transformer with the tied language modeling head on top (fully pre-trained)
- `gpt2DoubleHeadsModel`: OpenAI GPT-2 Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT-2 Transformer is pre-trained, the multiple choice classification head is only initialized and has to be trained)

Note that two versions of GPT-2 are available for use: the small version (`gpt2`: English model with 12-layer, 768-hidden, 12-heads, 117M parameters) and the medium version (`gpt2-medium`: English model with 24-layer, 1024-hidden, 16-heads, 345M parameters).

### Requirements

Unlike most other PyTorch Hub models, GPT requires a few additional Python packages to be installed.

```bash
pip install tqdm boto3 requests regex
```

Using `python3` is recommended to use these models especially regarding the use of the tokenizer.

### Example

Here is an example on how to tokenize the text with `gpt2Tokenizer`, and then get the hidden states computed by `gpt2Model` or predict the next token using `gpt2LMHeadModel`. Finally, we showcase how to use `gpt2DoubleHeadsModel` to combine the language modeling head and a multiple choice classification head.

```python
### First, tokenize the input
#############################
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2Tokenizer', 'gpt2')

#  Prepare tokenized input
text_1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
tokens_tensor_1 = torch.tensor([indexed_tokens1])
tokens_tensor_2 = torch.tensor([indexed_tokens2])


### Get the hidden states computed by `gpt2Model`
#################################################
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2Model', 'gpt2')
model.eval()

# Predict hidden states features for each layer
# past can be used to reuse precomputed hidden state in a subsequent predictions
with torch.no_grad():
	hidden_states_1, past = model(tokens_tensor_1)
	hidden_states_2, past = model(tokens_tensor_2, past=past)


### Predict the next token using `gpt2LMHeadModel`
##################################################
lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2LMHeadModel', 'gpt2')
lm_model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
	predictions_1, past = lm_model(tokens_tensor_1)
	predictions_2, past = lm_model(tokens_tensor_2, past=past)

# Get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.decode([predicted_index])
assert predicted_token == ' who'


### Language modeling and multiple choice classification `gpt2DoubleHeadsModel`
###############################################################################
double_head_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'gpt2DoubleHeadsModel', 'gpt2')
double_head_model.eval() # Set the model to train mode if used for training

tokens_tensor = torch.tensor([[indexed_tokens1, indexed_tokens2]])
mc_token_ids = torch.LongTensor([[len(tokenized_text1)-1, len(tokenized_text2)-1]])

with torch.no_grad():
    lm_logits, multiple_choice_logits, presents = double_head_model(tokens_tensor, mc_token_ids)
```

### Resources

 - Paper: [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/)
 - [Blogpost from OpenAI](https://openai.com/blog/better-language-models/)
 - Initial repository (with detailed examples and documentation): [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
