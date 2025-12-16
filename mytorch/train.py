import random
import numpy as np
from mytorch.losses import *
from mytorch.arithmetics import *
from mytorch.models import *
from mytorch.optimizers import *
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

random.seed(0)

def diagonal(pts: int):
    xs = []
    for i in range(pts):
        xs.append([random.uniform(-1, 1), random.uniform(-1, 1)])
    ys = []
    for x in xs:
        if x[0] >= x[1]:
            ys.append(0)
        else:
            ys.append(1)
    return xs, ys

def circle(pts: int):
    xs = []
    for i in range(pts):
        xs.append([random.uniform(-1, 1), random.uniform(-1, 1)])
    ys = []
    for x in xs:
        if x[0] ** 2 + x[1] ** 2 >= 0.5:
            ys.append(1)
        else:
            ys.append(0)
    return xs, ys

def spiral(pts: int):
    n_class = pts // 2
    xs, ys = [], []

    for i in range(n_class):
        r = i / n_class
        theta = 2 * math.pi * r
        x1 = r * math.cos(theta) + random.gauss(0, 0.02)
        y1 = r * math.sin(theta) + random.gauss(0, 0.02)
        xs.append([x1, y1])
        ys.append(0)

    for i in range(n_class):
        r = i / n_class
        theta = 2 * math.pi * r + math.pi
        x2 = r * math.cos(theta) + random.gauss(0, 0.02)
        y2 = r * math.sin(theta) + random.gauss(0, 0.02)
        xs.append([x2, y2])
        ys.append(1)

    return xs, ys

def curves(epochs, losses, accs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.plot(range(1, epochs + 1), losses, color='k', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.set_title(f'Loss Curves')

    ax2 = axes[1]
    ax2.plot(range(1, epochs + 1), accs, color='k', label='Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='y')
    ax2.set_title(f'Accuracy Curves')

    plt.tight_layout()
    plt.show()

def train(model, dataset, optim: str, epochs: int, lr: float):
    if optim == 'Adam':
        optimizer = Adam(model.get_params(), lr)
    elif optim == 'Adagrad':
        optimizer = Adagrad(model.get_params(), lr)
    elif optim == 'SGD':
        optimizer = SGD(model.get_params(), lr)
    else:
        raise ValueError(f'Optimizer {optim} is not supported')

    xs_list, ys_list = dataset
    pts = len(ys_list)
    ys = Tensor(ys_list, (pts, 1))
    xs = Tensor([i for x in xs_list for i in x], (pts, 2))

    losses, accs = [], []

    for epoch in range(1, epochs + 1):
        prob = model(xs).sigmoid()
        loss = BCELoss('mean')(prob, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = prob > 0.5
        correct = pred == ys
        num_correct = correct.sum(0)
        acc = (num_correct / pts).storage[0] * 100

        losses.append(loss.storage[0])
        accs.append(acc)

        if epoch % 100 == 0:
            print(f"epoch: {epoch}/{epochs}, acc: {acc:.2f}%, loss: {loss.storage[0]:.4f}")

    xs_test_list, ys_test_list = dataset
    pts = len(ys_test_list)
    ys_test = Tensor(ys_test_list, (pts, 1))
    xs_test = Tensor([i for x in xs_test_list for i in x], (pts, 2))
    prob = model(xs_test).sigmoid()
    pred = prob > 0.5
    correct = pred == ys_test
    num_correct = correct.sum(0)
    print()
    print(f"test acc = {(num_correct / pts).storage[0] * 100 :.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    ax1 = axes[0]
    ax1.plot(range(1, epochs + 1), losses, color='k', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.set_title(f'Loss Curves')

    ax2 = axes[1]
    ax2.plot(range(1, epochs + 1), accs, color='k', label='Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='y')
    ax2.set_title(f'Accuracy Curves')

    ax3 = axes[2]
    grid_size = 200
    x1 = np.linspace(-1, 1, grid_size)
    x2 = np.linspace(-1, 1, grid_size)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.stack([xx1.flatten(), xx2.flatten()], axis=1)

    grid_tensor = Tensor(grid.flatten().tolist(), (grid_size ** 2, 2))
    prob = model(grid_tensor).sigmoid()
    prob = np.array(prob.storage).reshape(grid_size, grid_size)

    ax3.contourf(xx1, xx2, (prob > 0.5), levels=[0, 0.5, 1], colors=['deepskyblue', 'lightcoral'])

    xs0 = [x for x, y in zip(xs_list, ys_list) if y == 0]
    xs1 = [x for x, y in zip(xs_list, ys_list) if y == 1]
    ax3.scatter([x[0] for x in xs0], [x[1] for x in xs0], color='blue', label='Class 0')
    ax3.scatter([x[0] for x in xs1], [x[1] for x in xs1], color='red', label='Class 1')

    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_title('Decision Boundary')
    ax3.legend(bbox_to_anchor=(1, 1))
    ax3.axis('equal')
    plt.tight_layout()
    plt.show()


def mnist(batch_size: int = 64, train: bool = True, download: bool = True):
    dataset = datasets.MNIST(root='./data', train=train, download=download, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size)

    batched = []
    for i, (images, labels) in enumerate(dataloader):
        if i >= 10:
            break
        batched.append([Tensor(images.flatten().numpy().tolist(), (images.shape[0], 1, 28, 28)),
                        Tensor(labels.numpy().tolist())])

    return batched

def train_cnn(model, optim: str):
    epochs = 10
    batch_size = 64

    if optim == 'Adam':
        optimizer = Adam(model.get_params(), lr = 0.001) # Adam optimizer is much better and more stable than SGD optimizer
    elif optim == 'Adagrad':
        optimizer = Adagrad(model.get_params(), lr = 0.1)
    elif optim == 'SGD':
        optimizer = SGD(model.get_params(), lr=0.01)
    else:
        raise ValueError(f'Optimizer {optim} is not supported')

    batched = mnist(batch_size=batch_size, train=True)
    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_correct = 0
        for images, labels in batched:
            prob = model(images).reshape(batch_size, 10)
            prob = prob.softmax(-1)
            loss = CrossEntropyLoss('mean')(prob, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = prob.max(1, keepdim=False)
            correct = pred == labels
            num_correct = correct.sum(0) # type: ignore
            total_loss += loss.storage[0]
            total_correct += num_correct
        loss = total_loss / len(batched)
        losses.append(loss)
        acc = (total_correct / (batch_size * len(batched))).storage[0] * 100
        accs.append(acc)

        print(f"epoch: {epoch}/{epochs}, "
              f"acc: {acc:.2f}%, "
              f"loss: {loss:.4f}")
    curves(epochs, losses, accs)