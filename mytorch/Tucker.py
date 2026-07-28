import string
import torch
import torch.nn as nn
import torch.optim as optim
from mytorch.train import curves, circle, spiral
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

class TuckerModel(nn.Module):
    """
    Tucker model, the generalization of the CP model (section 3.2 of the report) obtained by replacing the
    diagonal core of the CP decomposition with a full, TRAINABLE core tensor, as described in section 3.1.2.

    Attributes:
        N (int): The number of vectorized patches of one input instance (e.g. image or 2D point),
        M (int): The number of representation functions, consist with the number of input channels in CNN.
        Y (int): The number of output classes, consist with the number of output channels in CNN.
        ranks (list): A list of length N, the Tucker-rank R_i of each mode i, consist with the number of
        intermediate channels in CNN.
    """

    def __init__(self, N: int, s: int, M: int, Y: int, ranks: list):
        super(TuckerModel, self).__init__()
        if (not isinstance(N, int) or not isinstance(s, int) or not isinstance(M, int) or not isinstance(Y, int)
                or not all(isinstance(r, int) for r in ranks)):
            raise TypeError("N, s, M, Y and ranks must be integers.")
        if N < 1 or s < 1 or M < 1 or Y < 1 or not all(r > 1 for r in ranks):
            raise ValueError("N, s, M, Y and ranks must all be positive, and every rank must exceed 1.")
        if len(ranks) != N:
            raise ValueError(f"The number of ranks must equal to N = {N}, one rank R_i per mode.")

        self.N = N
        self.s = s
        self.M = M
        self.Y = Y
        self.ranks = ranks

        # The factor matrices U^(i) in R^{M x R_i}, one per patch i, playing the same role as the unshared
        # 1x1 convolution kernels a^(z,i) of the CP model
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(M, r) / np.sqrt(M)) for r in ranks])
        # The full core tensor G^(y) of shape (Y, R_1, ..., R_N), reduces to the CP model's diagonal core
        # when restricted to its diagonal entries
        self.core = nn.Parameter(torch.randn(Y, *ranks) / np.sqrt(np.prod(ranks)))

        # The representation layer which converts the shape of input from (N, s) to (N, M), identical in
        # structure to the one used in the HT model
        self.repr = nn.Sequential(nn.Linear(s, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, M))

        # Build the einsum equation once, e.g. for N=2: 'br,bs,yrs->by'
        letters = [c for c in string.ascii_lowercase if c not in ('b', 'y')][:N]
        self._mode_letters = letters
        factor_terms = ','.join(f'b{c}' for c in letters)
        core_term = 'y' + ''.join(letters)
        self._equation = f'{factor_terms},{core_term}->by'

    def forward(self, X):
        """
        Forward pass of the Tucker model to compute the scores of each output class, supporting batch
        optimization.

        Args:
            X: Processed input instance of size (batch_size, N, s).

        Returns:
            Output scores of each class, output of size (batch_size, Y).
        """
        F = self.repr(X) # From (batch_size, N, s) to (batch_size, N, M)

        # C^(i) = F_{:,i,:} @ U^(i), the mode-i product between the representation and the i-th factor matrix
        Cs = [F[:, i, :] @ self.factors[i] for i in range(self.N)] # Each of size (batch_size, R_i)

        # Contract every C^(i) with the core tensor G^(y) in one shot, e.g. for N=2:
        # o_{b,y} = sum_{r,s} C^(1)_{b,r} * C^(2)_{b,s} * G_{y,r,s}
        return torch.einsum(self._equation, *Cs, self.core) # (batch_size, Y)

def make_dataset(task: str, pts: int):
    if task == 'circle':
        xs_list, ys_list = circle(pts)
    elif task == 'spiral':
        xs_list, ys_list = spiral(pts)
    else:
        raise ValueError(f"Unknown task '{task}', choose 'circle' or 'spiral'.")
    xs = torch.tensor(xs_list, dtype=torch.float32).unsqueeze(-1) # (pts, N=2, s=1)
    ys = torch.tensor(ys_list, dtype=torch.long)
    return xs, ys, xs_list, ys_list

def plot_boundary(model, xs_list, ys_list, device, title):
    grid_size = 200
    x1 = np.linspace(-1.2, 1.2, grid_size)
    x2 = np.linspace(-1.2, 1.2, grid_size)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.stack([xx1.flatten(), xx2.flatten()], axis=1)
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(-1).to(device) # (grid_size ** 2, N=2, s=1)

    model.eval()
    with torch.no_grad():
        pred = model(grid_tensor).argmax(dim=1).cpu().numpy().reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx1, xx2, pred, levels=[-0.5, 0.5, 1.5], colors=['deepskyblue', 'lightcoral'])

    xs0 = [x for x, y in zip(xs_list, ys_list) if y == 0]
    xs1 = [x for x, y in zip(xs_list, ys_list) if y == 1]
    ax.scatter([x[0] for x in xs0], [x[1] for x in xs0], color='blue', s=10, label='Class 0')
    ax.scatter([x[0] for x in xs1], [x[1] for x in xs1], color='red', s=10, label='Class 1')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

def train_tucker(task: str = 'circle', epochs: int = 100, dataset=None, plot: bool = True):
    """
    Train the Tucker model on a 2D binary classification task ('circle' or 'spiral') and visualize the result.

    Args:
        task (str): Either 'circle' or 'spiral', selects which 2D dataset to train on.
        epochs (int): The number of training epochs.
        dataset: Optional pre-generated (xs, ys, xs_list, ys_list) tuple, as returned by make_dataset(). Lets
        multiple models be trained and fairly compared on the exact same data. If None, a fresh dataset is
        generated internally.
        plot (bool): Whether to plot the loss/accuracy curves and the decision boundary after training.

    Returns:
        losses, accs, model, xs_list, ys_list: The per-epoch training loss and accuracy, the trained model,
        and the raw dataset used for training.
    """
    # Hyperparameters
    N = 2 # Each 2D point is treated as N=2 patches, one per coordinate
    s = 1 # Every patch is a single scalar coordinate
    M = 8
    Y = 2
    ranks = [8, 8] # Same total budget as the CP model's Z=16 rank-1 terms (R_1 * R_2 = 64 core entries)
    pts = 1000
    learning_rate = 0.01

    # Define the model, choose the loss function and optimizer
    model = TuckerModel(N, s, M, Y, ranks)
    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # Cross entropy loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

    xs, ys, xs_list, ys_list = dataset if dataset is not None else make_dataset(task, pts)
    xs, ys = xs.to(device), ys.to(device)

    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        output = model(xs)
        loss = criterion(output, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        acc = 100 * predicted.eq(ys).sum().item() / len(ys_list)

        losses.append(loss.item())
        accs.append(acc)
        if epoch % 10 == 0:
            print(f'epoch: {epoch}/{epochs}, loss: {loss.item():.4f}, acc: {acc:.2f}%')

    if plot:
        curves(epochs, losses, accs)
        plot_boundary(model, xs_list, ys_list, device, f'Tucker Model Decision Boundary ({task})')
    return losses, accs, model, xs_list, ys_list
