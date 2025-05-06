import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from torchvision import datasets, transforms

from Dataprocessor import DataProcessor
from Evaluate import evaluate
from MLP import MLP
from Train import train
from Visualize import visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mnist_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # chuẩn hóa tốt hơn
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    return dataset


class PyTorchMLP(BaseEstimator):
    def __init__(self, hidden_layers=[512], lr=0.001,dropout = 0.0):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.model = None
        self.dropout = dropout

    def fit(self, X, y):
        input_size = X.shape[1]
        self.model = MLP(input_size=input_size, hidden_layers=self.hidden_layers, output_size=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X).float().to(device)
        y_tensor = torch.tensor(y).long().to(device)

        for epoch in range(1, 200):  # tăng số epoch lên
            self.model.train()
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X).float().to(device)
        with torch.no_grad():
            output = self.model(X_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy()

    def score(self, X, y):
        predictions = self.predict(X)
        return (predictions == y).mean()


# Load và xử lý dữ liệu
processor = DataProcessor(batch_size=64)

train_dataset, test_dataset = processor.load_and_process_data(
    source_type='custom',
    loader_func=mnist_loader,
    test_size=0.2
)

# Lấy dữ liệu từ Subset
X_train = train_dataset.dataset.data[train_dataset.indices].numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = train_dataset.dataset.targets[train_dataset.indices].numpy()

X_test = test_dataset.dataset.data[test_dataset.indices].numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = test_dataset.dataset.targets[test_dataset.indices].numpy()

# Tìm tham số tốt nhất
param_dist = {
    'hidden_layers': [[512], [512, 256], [512, 256, 128]],
    'dropout' :[0.1,0.2,0.5],
    
    'lr': [0.001, 0.0001]
}

model = PyTorchMLP()

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, verbose=2)
random_search.fit(X_train, y_train)

print("Tham số tốt nhất: ", random_search.best_params_)

best_model = random_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Độ chính xác trên tập kiểm tra với tham số tốt nhất: {test_accuracy * 100:.2f}%")

# Tạo DataLoader để trực quan
test_loader = processor.create_dataloader(test_dataset, shuffle=False)
#visualize(best_model.model, test_loader, device=device, num_images=12)
