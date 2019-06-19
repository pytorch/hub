---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Transformer-XL
summary:  Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
category: researchers
image: huggingface-logo.png
author: HuggingFace Team
tags: [nlp]
github-link: https://github.com/huggingface/pytorch-pretrained-BERT.git
featured_image_1: no-image
featured_image_2: no-image
---

### Model Description

Transformer-XL was released together with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860) by Zihang Dai, et al. This PyTorch implementation of Transformer-XL is an adaptation of the original [PyTorch implementation](https://github.com/kimiyoung/transformer-xl) which has been slightly modified to match the performances of the TensorFlow implementation and allow to re-use the pretrained weights.

Here are two models based on the author's pre-trained weights along with the associated Tokenizer.
It includes:
- `transformerXLModel`: Transformer-XL model which outputs the last hidden state and memory cells (fully pre-trained)
- `transformerXLLMHeadModel`: Transformer-XL with the tied adaptive softmax head on top for language modeling which outputs the logits/loss and memory cells (fully pre-trained)

### Requirements

Unlike most other PyTorch Hub models, Transformer-XL requires a few additional Python packages to be installed.

```bash
pip install tqdm boto3 requests regex
```

### Example

Here is an example on how to tokenize the text with `transformerXLTokenizer`, and then get the hidden states computed by `transformerXLModel` or predict the next token using `transformerXLLMHeadModel`.

```python
### First, tokenize the input
#############################
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'transformerXLTokenizer', 'transfo-xl-wt103')

#  Prepare tokenized input
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

### Get the hidden states computed by `transformerXLModel`
##########################################################
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'transformerXLModel', 'transfo-xl-wt103')
model.eval()

# Predict hidden states features for each layer
# past can be used to reuse precomputed hidden state in a subsequent predictions
with torch.no_grad():
	hidden_states_1, mems_1 = model(tokens_tensor_1)
	hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

### Predict the next token using `transformerXLLMHeadModel`
###########################################################
lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'transformerXLLMHeadModel', 'transfo-xl-wt103')
lm_model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
	predictions_1, mems_1 = lm_model(tokens_tensor_1)
	predictions_2, mems_2 = lm_model(tokens_tensor_2, mems=mems_1)

# Get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'who'
```

### Resources

 - Paper: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860)
 - Initial repository (with detailed examples and documentation): [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
 - Original author's [implementation](https://github.com/kimiyoung/transformer-xl) 
