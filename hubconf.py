import torch
from src.ToqiNet import ToqiNet

dependencies = ['torch']

def toqinet(pretrained=False, **kwargs):

    model = ToqiNet(**kwargs)
    if pretrained:
        print("Pre-trained weights not available. Model will be loaded without pre-trained weights.")
    
    return model

# Load the model without pre-trained weights
model = toqinet(pretrained=False)
print(model)


