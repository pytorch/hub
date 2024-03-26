import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from ToqiNet import ToqiNet
from CustomDataset import CustomDataset
       
"""
in here i have set the condition to set GPU for default for training approch 

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = (3, 256, 256)  
num_classes = 2 

model = ToqiNet(num_classes=2)
model.to(model.device)

train_dataset = CustomDataset(root_dir='dataset/training_set', transform=model.transform)
test_dataset = CustomDataset(root_dir='dataset/test_set', transform=model.transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

total_params = model.count_parameters()
print(f"Total number of parameters in the model: {total_params}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

"""
in this section the epochs and training proggress is defined (using tqdm)

"""

num_epochs = 100
print_every = 5  
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))  
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (i + 1):.4f}") 

    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'class_labels': train_dataset.data.class_to_idx
}, 'Image classification\model\ToqiNet6.pt')

with open('Image classification\model\class_labels.txt', 'w') as f:
    for class_name, class_idx in train_dataset.data.class_to_idx.items():
        f.write(f'{class_name}: {class_idx}\n')

model.eval()
correct = 0
total = 0

class_correct = defaultdict(int)
class_total = defaultdict(int)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            class_correct[pred] += int(pred == label)
            class_total[label] += 1

accuracy = 100 * correct / total
print('Accuracy of the network on the test images: %.2f %%' % accuracy)

for class_name, class_idx in test_dataset.data.class_to_idx.items():
    print(f'Class: {class_name}, Total Images: {class_total[class_idx]}')


