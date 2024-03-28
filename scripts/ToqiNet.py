"""
Hello everyone, Hope you are enjoying the day . As you have access the model. Let me tell you that this model is in development rigt now
its not the final version. Its inspired by the AlexNet but it has larger parametar then AlexNet although the accurecy is not decent 
right now. Still its much more simplier then any other model available. This model also provide you the Coustomdataset 
feature which implies to autoresize and image size handeling while training image.

COPYRIGHT RESERVED BY : thameedtoqi123@gmail.com (C) 2024


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms




class ToqiNet(nn.Module):
    def __init__(self, num_classes=2, fine_tuning=True, weight_decay=0.6):
        super(ToqiNet, self).__init__()
        self.fine_tuning = fine_tuning
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.features_output_shape = self._calculate_conv_output_shape()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.features_output_shape, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

        """
        i have used SGD for optimaizer thus, the  model is little bit complex  and to put less stress on training
        """
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_fine_tuning(self.fine_tuning)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.transform(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    """
    in here i have defined all the function to work in the ToqiNet class
    """

    def count_parameters(self):                                             
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def set_dropout_rate(self, rate):
        for layer in self.classifier:
            if isinstance(layer, nn.Dropout):
                layer.p = rate

    def set_fine_tuning(self, enable):
        for param in self.features.parameters():
            param.requires_grad = enable

    def _calculate_conv_output_shape(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            x = self.features(x)
            return x.size(1) * x.size(2) * x.size(3)



model = ToqiNet(num_classes=2)
model.to(model.device)

