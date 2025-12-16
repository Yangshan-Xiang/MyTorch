class Optimizer:
    """
    The fundamental construction for an optimizer, will be inherited by actual optimizers.
    """

    def __init__(self, params: list, lr: float, maximize: bool, weight_decay: float):
        self.params = params
        self.lr = lr
        self.maximize = maximize
        self.weight_decay = weight_decay

    def zero_grad(self, none_grad: bool = True) -> None:
        for param in self.params:
            if none_grad:
                param.grad = None
            else:
                param.grad = 0

class SGD(Optimizer):
    """
    The Stochastic Gradient Descent optimizer. It is the most classic, well-known and fundamental optimizer.

    """

    def __init__(self, params: list, lr: float = 0.001, maximize: bool = False, weight_decay: float = 0,
                 momentum: float = 0, dampening: float = 0, nesterov: bool = False):
        super().__init__(params, lr, maximize, weight_decay)
        self.momentum = momentum  # Use the momentum to utilize the gradient calculated in last step.
        self.dampening = dampening
        self.nesterov = nesterov
        self.buffer = [None] * len(params)  # Store necessary values when SGD needs momentum

    def step(self) -> None:
        for i, param in enumerate(self.params):
            if self.maximize:
                grad = -1 * param.gradient
            else:
                grad = param.gradient
            if self.weight_decay != 0:
                grad += self.weight_decay * param.constant()
            else:
                pass
            if self.momentum != 0:
                if self.buffer[i] is None:
                    self.buffer[i] = grad
                else:
                    self.buffer[i] = self.momentum * self.buffer[i] + (1 - self.dampening) * grad  # type: ignore
                if self.nesterov:
                    grad += self.momentum * self.buffer[i]  # type: ignore
                else:
                    grad = self.buffer[i]
            else:
                pass
            param.update(param.constant() - self.lr * grad)

class Adam(Optimizer):
    """
    The Adam optimizer. It is very stable and smooth in most training occasions, the go-to choice for most people.
    """

    def __init__(self, params: list, lr: float = 0.001, maximize: bool = False, weight_decay: float = 0,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8, amsgrad: bool = False):
        super().__init__(params, lr, maximize, weight_decay)
        self.betas = betas
        self.eps = eps  # For numerical stability
        self.amsgrad = amsgrad
        self.t = 0  # Keep track of which step we are at
        self.m = [0] * len(params)  # The 1st moment
        self.v = [0] * len(params)  # The 2nd moment
        self.v_max = [0] * len(params)  # The maximum between the current 2nd moment and the 2nd moment from last step,
        # only needed when amsgrad is True.

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            if self.maximize:
                grad = -1 * param.gradient
            else:
                grad = param.gradient
            if self.weight_decay != 0:
                grad += self.weight_decay * param.constant()
            else:
                pass
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * grad ** 2
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            if self.amsgrad:
                self.v_max[i] = max(self.v_max[i], self.v[i])
                v_hat = self.v_max[i] / (1 - self.betas[1] ** self.t)
            else:
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            param.update(param.constant() - self.lr * m_hat / (v_hat.sqrt() + self.eps))

class Adagrad(Optimizer):
    """
    The Adagrad optimizer. It often requires larger learning rate like 0.1 to achieve a rather good performance.
    """

    def __init__(self, params: list, lr: float = 0.1, maximize: bool = False, weight_decay: float = 0,
                 lr_decay: float = 0, initial: float = 0, eps: float = 1e-8):
        super().__init__(params, lr, maximize, weight_decay)
        self.lr_decay = lr_decay
        self.state_sum = [initial] * len(params)  # Used to accumulate the squared gradient from previous steps.
        self.eps = eps  # For numerical stability
        self.t = 1

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.params):
            grad = param.gradient
            lr_tilde = self.lr / (1 + (self.t - 1) * self.lr_decay)
            if self.weight_decay != 0:
                grad += self.weight_decay * param.constant()
            else:
                pass
            self.state_sum[i] += grad ** 2
            param.update(param.constant() - lr_tilde * grad / (self.state_sum[i].sqrt()+ self.eps)) # type: ignore