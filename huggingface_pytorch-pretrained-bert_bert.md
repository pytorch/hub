---
layout: hub_detail
background-class: hub-background
body-class: hub
title: BERT
summary:  Bidirectional Encoder Representations from Transformers.
category: researchers
image: huggingface-logo.png
author: HuggingFace Team
tags: [NLP, BERT]
github-link: https://github.com/huggingface/pytorch-pretrained-BERT.git
featured_image_1: no-image
featured_image_2: no-image
---

### Model Description

BERT was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin et al. The model is based on the Transformer architecture introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani et al and has led to significant improvements on a wide range of downstream tasks.

Here are 8 models based on BERT with [Google's pre-trained models](https://github.com/google-research/bert) along with the associated Tokenizer.
It includes:
- `bertTokenizer`: perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization
- `bertModel`: raw BERT Transformer model (fully pre-trained)
- `bertForMaskedLM`: BERT Transformer with the pre-trained masked language modeling head on top (fully pre-trained)
- `bertForNextSentencePrediction`: BERT Transformer with the pre-trained next sentence prediction classifier on top (fully pre-trained)
- `bertForPreTraining`: BERT Transformer with masked language modeling head and next sentence prediction classifier on top (fully pre-trained)
- `bertForSequenceClassification`: BERT Transformer with a sequence classification head on top (BERT Transformer is pre-trained, the sequence classification head is only initialized and has to be trained)
- `bertForMultipleChoice`: BERT Transformer with a multiple choice head on top (used for task like Swag) (BERT Transformer is pre-trained, the multiple choice classification head is only initialized and has to be trained)
- `bertForTokenClassification`: BERT Transformer with a token classification head on top (BERT Transformer is pre-trained, the token classification head is only initialized and has to be trained)
- `bertForQuestionAnswering`: BERT Transformer with a token classification head on top (BERT Transformer is pre-trained, the token classification head is only initialized and has to be trained)


### Example

Here is an example on how to tokenize the input text with `bertTokenizer`, and then get the hidden states computed by `bertModel` or predict masked tokens using `bertForMaskedLM`.

```python
### First, tokenize the input
import torch
tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)

# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


### Get the hidden states computed by `bertModel`
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-cased')
model.eval()

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)


### Predict masked tokens using `bertForMaskedLM`
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

maskedLM_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForMaskedLM', 'bert-base-cased')
maskedLM_model.eval()

with torch.no_grad():
    predictions = maskedLM_model(tokens_tensor, segments_tensors)

# Get the predicted token
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'Jim'
```


### Resources

 - Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
 - Initial repository (with detailed examples and documentation): [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
