
"""
version --1.1.2
Hello everyone, Hope you are enjoying the day . As you have access the model. Let me tell you that this model is in development rigt now
its not the final version. in this new version i have worked on the accurecy on a single image after training . Still its much more simplier then any other model available. This model also provide you the Coustomdataset 
feature which implies to autoresize and image size handeling while training image.

COPYRIGHT RESERVED BY : thameedtoqi123@gmail.com (C) 2024


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS


class ToqiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.data = ImageFolder(self.root_dir, transform=self.transform)
        self.class_to_idx = self._find_classes(root_dir)
        self.num_classes = len(self.class_to_idx)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx
    
    def _find_images(self):
        images = []
        for filename in os.listdir(self.root_dir):
            if self._is_image_file(filename):
                images.append(os.path.join(self.root_dir, filename))
        return images

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



class ToqiNet(nn.Module):
    def __init__(self, num_classes=None, fine_tuning=True, weight_decay=0.6):
        super(ToqiNet, self).__init__()
        self.fine_tuning = fine_tuning

     
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
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
        
        # Set default num_classes if None, Its for the error handeling 
        if num_classes is None:
            num_classes = 2  # Assuming binary classification , its a imaginary number 
        
        self.num_classes = num_classes

    
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.features_output_shape, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_classes),
        )

        self.similarity_threshold = 0.6

        self.optimizer = optim.SGD(self.parameters(), lr=0.001, weight_decay=weight_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
        self.set_fine_tuning(self.fine_tuning)


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
        
    def set_dataset_root(self, root_dir):
        self.transform.root_dir = root_dir    
        
    def classify_image(self, image):
        with torch.no_grad():
            output = self(image)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            class_confidence = probabilities[0, predicted].item()
            
            similarity_percentage = self.calculate_similarity(image)
            
            if class_confidence >= self.similarity_threshold:
                return predicted.item(), 1.0
            elif class_confidence >= 0.5:
                adjusted_confidence = 0.95 + 0.02 * similarity_percentage
                return predicted.item(), min(1.0, adjusted_confidence)
            else:
                return None, None
    
    def calculate_similarity(self, image):
        with torch.no_grad():
            features = self.features(image)
            flattened_features = torch.flatten(features)
            learned_parameters = torch.flatten(torch.cat([param.view(-1) for param in self.parameters()]))
            similarity = torch.cosine_similarity(flattened_features, learned_parameters, dim=0)
            return similarity.item()   



model = ToqiNet(num_classes=None)
model.to(model.device)
