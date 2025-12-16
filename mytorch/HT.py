import torch
import torch.nn as nn
import torch.optim as optim
from mytorch.train import curves
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(0)

# The key to implement this model is how to index and structure all the parameters such that they can
# be computed recursively
class HTModel(nn.Module):
    """
    HT model depicted in the original paper with TRAINABLE parameters.

    Attributes:
        N (int): The number of vectorized patches of one input instance (e.g. image),
        need to be the power of 2 for simplicity as described in the paper.
        M (int): The number of representation functions, consist with the number of input channels in CNN.
        Y (int): The number of output classes, consist with the number of output channels in CNN.
        ranks (list): A list of ranks of the HT decomposition, consist with the number of intermediate channels in CNN.
    """

    def __init__(self, N:int, s:int, M:int, Y:int, ranks:list):
        super(HTModel, self).__init__()
        # The notations are consistent with the notations used in the original paper
        if (not isinstance(N, int) or not isinstance(s, int) or not isinstance(M, int) or not isinstance(Y, int)
                or not all(isinstance(r, int) for r in ranks)):
            raise TypeError("N, M, ranks and Y must be integers.")
        if N < 1 or s < 1 or M < 1 or Y < 1 or not all(r > 1 for r in ranks):
            raise ValueError("N, s, M, Y and ranks must all be positive.")
        if (N & (N - 1)) != 0:
            raise ValueError("N must be the power of 2.")
        if len(ranks) != np.log2(N):
            raise ValueError(f"The number of ranks must equal to {int(np.log2(N))}.")

        self.N = N
        self.s = s
        self.M = M
        self.Y = Y
        self.ranks = ranks

        self.L = int(np.log2(N)) # The number of hidden-layers, exclude the representation layer and output layer

        # Define the parameters
        self.params = nn.ParameterList()
        self.params.append(nn.Parameter(torch.randn(N, ranks[0], M))) # The parameters of the 1st hidden-layer
        for l in range(1, self.L):
            self.params.append(nn.Parameter(torch.randn(N // 2 ** l, ranks[l], ranks[l-1]))) # The parameters of the
            # rest hidden-layers
        self.params.append(nn.Parameter(torch.randn(Y, ranks[-1]))) # The parameters of the output layer.

        # The representation layer which converts the shape of input from (N, s) to (N, M), different from
        # the original paper, here the representation layer is trainable
        self.repr = nn.Sequential(nn.Linear(s, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, M))

    def forward(self, X):
        """
        Forward pass of the HT model to compute the scores of each output class, supporting batch optimization.

        Args:
            X: Processed input instance of size (batch_size, N, s).

        Returns:
            Output scores of each class, output of size (batch_size, Y).
        """
        F = self.repr(X) # From (batch_size, N, s) to (batch_size, N, M)

        # The task here is to compute the phi tensors hierarchically according to the equation (4) in the paper
        phis = [self.params[0]] # phi_0 tensor is just the stack of parameter vectors in the 1st hidden-layer
        for l in range (1, self.L):
            phi = torch.zeros((self.N // 2 ** l, self.ranks[l], self.M ** 2 ** l)) # Compute the intermediate level
            # phi tensors hierarchically
            # Compute the phi_l tensor by computing each phi_l_j_gamma recursively
            for j in range(self.N // 2 ** l):
                for gamma in range(self.ranks[l]):
                    for alpha in range(self.ranks[l-1]):
                        # This line of code EXACTLY matches the expression of phi_l_j_gamma depicted in
                        # the equation (4)
                        phi[j, gamma] += (torch.kron(phis[l-1][2 * j, alpha], phis[l-1][2 * j + 1, alpha]) *
                                          self.params[l][j, gamma, alpha])
            phis.append(phi)
        # Compute the final coefficient tensor A_y using the phi tensors
        A = torch.zeros((self.Y, * [self.M] * self.N)) # Every A_y tensor has the shape of M ^ N
        for y in range(self.Y):
            for alpha in range(self.ranks[-1]):
                # Again this EXACTLY matches the expression of A_y depicted in the equation (4)
                A[y] += (torch.kron(phis[-1][0, alpha], phis[-1][1, alpha]).reshape([self.M] * self.N)
                         * self.params[-1][y, alpha]) # Reshaped to M ^ N

        # Compute the score function h_y(X) depicted in function (2) in the paper
        batch_size = X.shape[0]
        h = torch.zeros((batch_size, self.Y))
        # Prepare the factor which can be directly multiplied with tensor A_y
        factor = torch.ones((batch_size, * A[0].shape))
        for b in range(batch_size):
            for i in range(self.N):
                factor[b] *= F[b][i, np.indices(A[0].shape)[i]] # np.indices() works perfectly with torch tensors

        for b in range(batch_size):
            for y in range(self.Y):
                h[b][y] = torch.sum(A[y] * factor[b]) # Multiply A_y with the factor and sum up all the elements

        return h

def train_ht():
    """
    Train the HT model on MNIST dataset and test the model. LOW accuracy are expected due to structure constraints.
    For example, the transformation of input image from shape (28, 28) to the required shape (N, s) by splitting
    and vectorization can damage the spatial information of the original image.

    """
    # Hyperparameters
    N = 4
    s = int(28 * 28 / N)
    M = 2
    Y = 10
    ranks = [16, 8]
    batch_size = 64
    learning_rate = 0.001
    epochs = 100

    # Define the model, choose the loss function and optimizer
    model = HTModel(N, s, M, Y, ranks)
    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # Cross entropy loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

    # Load and normalize the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    # Initialize the dataloaders
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        num_batches = 20 # Due to time limit, we only use a part of the MNIST dataset
        for i, (images, labels) in enumerate(train_loader):
            if i >= num_batches:
                break
            # Be careful with the batch size of the LAST batch, might be different and cause error
            images, labels = images.to(device), labels.to(device)
            # Transform the images from shape (batch_size, 1, 28, 28) to the required shape (batch_size, N, s)
            images = images.squeeze(1).reshape(-1, N, s)

            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = output.max(1)
            total += labels.shape[0]
            correct += predicted.eq(labels).sum().item()
        loss = running_loss / num_batches
        losses.append(loss)
        acc = 100 * correct / total
        accs.append(acc)
        if epoch % 10 == 0:
            print(f'epoch: {epoch}/{epochs}, loss: {loss:.4f}, acc: {acc:.2f}%')
    curves(epochs, losses, accs)