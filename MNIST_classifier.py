import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MNISTDataset(Dataset):
    def __init__(self):
        y_csv = pd.read_csv('mnist_train.csv', usecols=['label'])
        cols = pd.read_csv('mnist_train.csv', nrows=1).columns
        x_csv = pd.read_csv('mnist_train.csv', usecols=cols[1:])
        self.y = torch.tensor(y_csv.values, dtype=torch.float32)
        self.x = torch.tensor(x_csv.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class MNISTDatasetTest(Dataset):
    def __init__(self):
        y_csv = pd.read_csv('mnist_test.csv', usecols=['label'])
        cols = pd.read_csv('mnist_test.csv', nrows=1).columns
        x_csv = pd.read_csv('mnist_test.csv', usecols=cols[1:])
        self.y = torch.tensor(y_csv.values, dtype=torch.float32)
        self.x = torch.tensor(x_csv.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 10)

    def forward(self, x):
        x = functional.relu(self.l1(x))
        x = functional.relu(self.l2(x))
        x = self.l3(x)
        return functional.softmax(x, dim=1)


samples_in_batch = 100
MNIST_data = MNISTDataset()
MNIST_loader = DataLoader(dataset=MNIST_data, batch_size=samples_in_batch, shuffle=True)

MNIST_data_test = MNISTDatasetTest()
MNIST_loader_test = DataLoader(dataset=MNIST_data_test)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
epochs = 50

for epoch in range(epochs):
    for X, y in MNIST_loader:
        optimizer.zero_grad()
        output = net(X)

        one_hot = torch.zeros(samples_in_batch, 10)
        for i in range(len(one_hot)):
            one_hot[i][int(y[i])] = 1

        loss = functional.mse_loss(output, one_hot)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in MNIST_loader_test:
            output = net(X)
            if torch.argmax(output) == y:
                correct += 1
            total += 1

    print(f"Accuracy: {correct / total * 100:.1F}%")
