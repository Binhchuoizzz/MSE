import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    """
        Train for one epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """
        Evaluate the model
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("pytorch-mnist-tutorial")

    mlflow.pytorch.autolog()
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    with mlflow.start_run(run_name="pytorch-mnist-tutorial"):
        mlflow.set_tags(
            {
                "owner": "dsteam",
                "algorithm": "pytorch",
                "dataset": "mnist",
                "version": "1",
            }
        )
        mlflow.log_params(
            {
                "model": "SimpleNN",
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "device": device,
            }
        )

        model = SimpleNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_metric("total_params", total_params)

        print('Starting training...')
        print(f'Total parameters: {total_params}')

        start_time = time.time()
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, 
                                                criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader,
                                           criterion, device)
            epoch_time = time.time() - start_time

            mlflow.log_metrics({
                "train_loss": train_loss, 
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "epoch_time": epoch_time
                })
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Training completed in {time.time() - start_time:.2f} seconds')

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
        )
        print(f'Run ID: {mlflow.active_run().info.run_id}')

if __name__ == "__main__":
    main()