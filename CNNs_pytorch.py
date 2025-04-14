import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train = pd.read_csv("C:/Users/dungv/Computer_Vision/train.csv")
test = pd.read_csv("C:/Users/dungv/Computer_Vision/test.csv")


Y_train = train["label"].values
X_train = train.drop("label", axis=1).values / 255.0
X_test = test.values / 255.0


X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
Y_train = torch.tensor(Y_train, dtype=torch.long)


dataset = TensorDataset(X_train, Y_train)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=250, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=250, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.25),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 256),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        return self.net(x)

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Acc: {100 * correct / total:.2f}%")
    
model.eval()
predictions = []
with torch.no_grad():
        X_test = test.values / 255.0
        X_test = torch.tensor(X_test,dtype = torch.float32).reshape(-1,1,28,28).to(device)

        outputs = model(X_test)
        _,predicted = torch.max(outputs.data,1)
        predictions = predicted.cpu().numpy()
result = pd.DataFrame({
        'ImageId' : np.arange(1,len(predictions) + 1),
        'Label' : predictions
})
result.to_csv("C:/Users/dungv/Computer_Vision/result.csv", index=False)
print("kq dc luu vao result.csv")


plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


