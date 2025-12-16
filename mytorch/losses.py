from mytorch.tensor import *

class BCELoss:
    """
    The binary cross entropy loss.
    """
    def __init__(self, reduction: str = None):
        self.reduction = reduction

    def __call__(self, x: Tensor, y: Tensor):
        return self.forward(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return (-(y * x.log() + (1 - y) * (1 - x).log())).sum(0) / x.shape[0]
        if self.reduction == 'sum':
            return (-(y * x.log() + (1 - y) * (1 - x).log())).sum(0)
        else:
            return -(y * x.log() + (1 - y) * (1 - x).log()) # type: ignore

class CrossEntropyLoss:
    """
    The cross entropy loss function.
    """
    def __init__(self, reduction: str = None):
        self.reduction = reduction

    def __call__(self, x: Tensor, y: Tensor):
        return self.forward(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Convert from class indices to one-hot encoded labels like from 5 to [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        labels = []
        for i in y.storage:
            label = [0] * x.shape[-1]
            label[i] = 1
            labels.extend(label)
        labels = Tensor(labels, x.shape)

        if self.reduction == 'mean':
            return (-x.log() * labels).sum(1).sum(0) / x.shape[0] # (1, 1)
        elif self.reduction == 'sum':
            return (-x.log() * labels).sum(1).sum(0) # (1, 1)
        else:
            return (-x.log() * labels).sum(1) # (batch_size, 1)