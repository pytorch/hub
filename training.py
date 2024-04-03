
"""
Hello everyone, Hope you are enjoying the day . As you have access the model. Let me tell you that this model is in development rigt now
its not the final version. Its inspired by the AlexNet, but it has larger parametar then AlexNet although the accurecy is not decent 
right now. Still its much more simplier then any other model available. This model also provide you the Coustomdataset 
feature which implies to autoresize and image size handeling while training image.

COPYRIGHT RESERVED BY : thameedtoqi123@gmail.com (C) 2024


"""



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.ToqiNet import ToqiNet, ToqiDataset


"""
In here we are transforming the image for its comapatibility befor testing.

"""
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""
There we are defining the training directory and lso the testing through directory for checking the accurecy.
ToqiDataset is optimazie for handeling the dataset and preprocess the data before sending it for training

"""
train_dataset = ToqiDataset(root_dir='your training dataset path', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ToqiDataset(root_dir='your training dataset path', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32)

"""
in here the dataset for training path must be given into the model 

"""
model = ToqiNet(num_classes=train_dataset.num_classes)
model.set_dataset_root('your training dataset path')
model.to(model.device)





criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(model.device)

"""
for test purpose i have taken a small size of epochs it can be customaizeable

"""
num_epochs = 25
print_every = 5  

"""
Looping through  all the inputed and preprocess value

"""
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')



# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (i + 1):.4f}")

    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model on the test dataset
evaluate_model(model, test_loader, criterion)

"""
in here i have tried to save the model and save the class along with it through "class_to_idx". this can be done by creating a separate .txt file to store the class


"""
torch.save({
    'model_state_dict': model.state_dict(),  # Save model parameters
    'class_to_idx': train_dataset.class_to_idx,  # Save class to index mapping
    'similarity_threshold': model.similarity_threshold  # Save similarity threshold
}, 'your output directory to save the file and also the file name ')
