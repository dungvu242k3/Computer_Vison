#relu
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from Dataprocessor import DataProcessor
from Evaluate import evaluate
from MLP import MLP
from Train import train
from Visualize import visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""label_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}"""


def mnist_loader():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    return dataset

processor = DataProcessor(batch_size=64)

train_dataset, test_dataset = processor.load_and_process_data(
    source_type='custom',
    loader_func= mnist_loader,
    test_size=0.2
)

train_loader = processor.create_dataloader(train_dataset)
test_loader = processor.create_dataloader(test_dataset, shuffle=False)

model = MLP(input_size=784, hidden_layers=[256, 128], output_size=10,  dropout=0.2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 11):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc * 100:.2f} | Test Loss = {test_loss:.4f}, Test Acc = {test_acc * 100:.2f}")

visualize(model, test_loader, device=device, num_images=12)