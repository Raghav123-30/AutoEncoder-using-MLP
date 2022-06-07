import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


class mnistEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ae = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        return self.ae(x)


# Hyperparameters

dim = 784
ae = mnistEncoder(dim=dim)
lr = 3e-4
batch_size = 32
optimizer = optim.Adam(ae.parameters(), lr=lr)
criterion = nn.MSELoss()
epochs = 5
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3087,))]
)
dataset = datasets.MNIST(root='data/', download=True, transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epochi in range(epochs):
    for batch_idx, (digit, _) in enumerate(loader):
        digit = digit.view(-1, 28*28*1)
        pred = ae(digit)
        loss = criterion(pred, digit)

        ae.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx % 500) == 0:
            print(f"{epochi+1}/{epochs}     loss{loss:10.5f}%")


torch.save(ae.state_dict(), "trained.pth")
