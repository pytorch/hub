"""
this customdataset is needed for handeling the input tensors and theirsize for multiplication along with their labels

"""
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = ImageFolder(root_dir, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data.imgs[idx]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label