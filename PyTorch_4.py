import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize(size = (180, 180)),
    transforms.ToTensor(),])

test_transforms = transforms.Compose([
    transforms.Resize(size = (180, 180)),
    transforms.ToTensor(),])

full_dataset = ImageFolder(root = r"C:\Users\hp\Downloads\First semester\Distributed Data Analytics\Tutorial questions\flower_photos")

# total_size = len(full_dataset)
# train_size = 0.8 * total_size
# test_size = total_size - train_size

train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

train_dataset.dataset.transform = train_transforms
test_dataset.dataset.transform = test_transforms

# image, label = train_dataset[0]

# print("Image shape:", image.shape)  # Output: torch.Size([3, 64, 64])
# print("Label:", label)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(full_dataset.classes)

model = nn.Sequential(
    nn.Conv2d(3,16, kernel_size=3, padding=1),
    nn.Softmax(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.Softmax(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.Softmax(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*22*22, 128),
    nn.Softmax(),
    nn.Linear(128, num_classes)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
model_eval = nn.CrossEntropyLoss()

epoch_losses = []
epoch_accuracies = []

for epoch in range(50):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        output = model(images)
        loss = model_eval(output, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        logit, train_label = torch.max(output, 1)
        total += labels.size(0)
        correct += (train_label == labels).sum().item()

    avg_loss = running_loss/len(train_loader)
    accuracy = 100 * (correct / total)

    epoch_losses.append(avg_loss)
    epoch_accuracies.append(accuracy)


plt.plot(epoch_losses)
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.grid()
plt.show()

model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        # loss = model_eval(pred_label, label)

        logit, pred_label = torch.max(output, 1)

        test_total += labels.size(0)
        test_correct += (pred_label == labels).sum().item()

test_accuracy = (test_correct/test_total)*100
print(f"Test Accuracy: {test_accuracy:.2f}%")