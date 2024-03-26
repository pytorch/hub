import torch
from src.ToqiNet import ToqiNet

dependencies = ['torch']

def toqinet(pretrained=False, **kwargs):
    """
    Loads the ToqiNet model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        **kwargs: Additional parameters to pass to the ToqiNet constructor.
        
    Returns:
        ToqiNet: The loaded ToqiNet model.
    """
    model = ToqiNet(**kwargs)
    if pretrained:
        print("Pre-trained weights not available. Model will be loaded without pre-trained weights.")
        # If you have pre-trained weights available and want to load them, uncomment and replace the following line with the appropriate URL
        # state_dict = torch.load('url_to_pretrained_weights')
        # model.load_state_dict(state_dict)
    return model

# Load the model without pre-trained weights
model = toqinet(pretrained=False)
print(model)


