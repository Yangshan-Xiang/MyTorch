import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mytorch.train import curves



torch.manual_seed(0)  # For reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)


class TTModel(nn.Module):
    """
    The tensor train model obtained by taking the tensor train decomposition of the model coefficient tensor.
    """

    def __init__(self, input_size=28, bond_dim=64, num_classes=10, rep_dim=32):
        super().__init__()

        # Representation function as an MLP
        self.rep = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, rep_dim)
        )

        self.G1 = nn.Parameter(torch.randn(rep_dim, bond_dim) / (rep_dim ** 0.5))  # The first core tensor
        self.Gt = nn.Parameter(torch.randn(bond_dim, rep_dim, bond_dim) / (
                    bond_dim * rep_dim) ** 0.5)  # The intermediate core tensor (shared)
        self.GN = nn.Parameter(
            torch.randn(bond_dim, rep_dim, num_classes) / (bond_dim * rep_dim) ** 0.5)  # The last core tensor

    def forward(self, x):
        # x: (batch, length, input_size)
        length = x.size(1)  # The number of time steps of the model
        rep = self.rep(x)  # rep: (batch, length, rep_dim)

        x1 = rep[:, 0, :]  # Pick the first rows of the original images as the input at time step-1
        # x1: (batch, rep_dim)
        hi = x1 @ self.G1  # Compute the first hidden state h1
        # hi: (batch, bond_dim)
        hi = hi / (hi.norm(dim=-1, keepdim=True) + 1e-8)  # Normalization

        for i in range(1, length - 1):
            xi = rep[:, i, :]  # Pick the i-th rows of the original images as the input at time step-i
            hi = torch.einsum("bi,ijk,bj->bk", hi, self.Gt, xi)  # hi: (batch, bond_dim)
            hi = hi / (hi.norm(dim=-1, keepdim=True) + 1e-8)  # Normalization

        xN = rep[:, length - 1, :]  # Pick the last rows of the original images as the input at the last time step
        out = torch.einsum("bi,ijk,bj->bk", hi, self.GN, xN)  # (batch, num_classes)
        return out  # out: (batch, num_classes)


model = TTModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_epoch():
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    for i, (x, y) in enumerate(train_loader):
        if i >= len(train_loader) // 10:  # We only use one-tenth of the training set for seed reason
            break

        x = x.squeeze(1).to(device)  # x: (B, 1, 28, 28) -> (B, 28, 28)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, 100 * correct / total


def train_tt():
    epochs = 100
    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch()
        losses.append(loss)
        accs.append(acc)

        if epoch % 10 == 0:
            print(f'epoch: {epoch}/{epochs}, loss: {loss:.4f}, acc: {acc:.2f}%')
    curves(epochs, losses, accs)
    return losses, accs