import math
from mytorch.tensor import *

import random

class Parameter(Tensor):
    """
    To register a tensor, so it can be recognized as a model parameter.
    """
    def __init__(self, tensor: Tensor):
        super().__init__(tensor.storage, tensor.shape, tensor.stride, tensor.offset)
        self.history = History() # Need history by default

class Model:
    """
    Includes some basic methods for building models.
    """
    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Model()"

    def forward(self, x):
        raise NotImplementedError("The forward method should be implemented within its subclass.")

    def get_params(self) -> list:
        """
        As models can have not only their own parameters but also submodels, this method can help us get
        all the parameters in one clean list which will be fed into our optimizers.
        """

        params = []
        for attr in self.__dict__.items():
            if isinstance(attr[1], Parameter):
                params.append(attr[1])
            elif isinstance(attr[1], Model):
                params.extend(attr[1].get_params())
            elif isinstance(attr[1], Models):
                for model in attr[1]:
                    params.extend(model.get_params())
            else:
                pass
        return params

class Models:
    """

    """
    def __init__(self, *models):
        for model in models:
            if not isinstance(model, (Model, Models)):
                raise AssertionError(f"Expected Model or Models, got {type(model).__name__} instead.")
        self.models = list(models)

    def __call__(self, x):
        """

        """
        for model in self.models:
            x = model(x)
        return x

    def __getitem__(self, index):
        return self.models[index]

    def append(self, x) -> None:
        self.models = Models(*self.models, x).models

    def __repr__(self):
        return f"Models({self.models})"

class Linear(Model):
    """
    The linear model.
    """
    def __init__(self, inp_size, out_size, needs_bias: bool = True):
        """
        Specify the linear model.

        Args:
            inp_size (int): The input size.
            out_size (int): The output size.
            needs_bias (bool): Whether you need the bias term.
        """
        super().__init__()

        if not isinstance(inp_size, int) or not isinstance(out_size, int):
            raise TypeError("Input size and output size must be integers.")
        self.inp_size = inp_size
        self.out_size = out_size
        self.needs_bias = needs_bias

        random.seed(42)  # Fix the seed for reproducibility
        # Uniform Kaiming Initialization
        a = (6 / inp_size) ** 0.5
        self.weight = Parameter(Tensor([random.uniform(-a, a) for _ in range(inp_size * out_size)],
                                       shape = (inp_size, out_size)))
        if needs_bias:
            self.bias = Parameter(Tensor([0] * out_size))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass which computes the output of the linear model given the input.
        """

        if x.shape[-1] != self.inp_size:
            raise ValueError("The size of the last dimension of the input must equal to "
                             "the input size of the linear model.")

        if self.needs_bias:
            return x @ self.weight + self.bias
        else:
            return x @ self.weight

    def __repr__(self):
        return f"Linear({self.inp_size}, {self.out_size}, needs_bias={self.needs_bias})"

class MLP(Model):
    """
    The multi-layer perceptron which is a stack of linear layers with nonlinear activation functions in between.
    """

    def __init__(self, inp_size: int, hid_size: int, out_size: int, n_hid: int = 1, needs_bias: bool = True):
        """
        Specify the multi-layer perceptron.

        Args:
            inp_size (int): The input size.
            hid_size (int): The hidden size.
            out_size (int): The output size.
            n_hid (int): The number of hidden layers.
            needs_bias (bool): Whether you need the bias term.
        """
        super().__init__()

        self.inp_size = inp_size
        self.out_size = out_size
        self.hid_size = hid_size
        self.n_hid = n_hid
        self.needs_bias = needs_bias

        self.inp_layer = Linear(inp_size, hid_size, needs_bias)
        self.hid_layer = Models()
        for _ in range(n_hid):
            self.hid_layer.append(Linear(hid_size, hid_size, needs_bias))
        self.out_layer = Linear(hid_size, out_size, needs_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.inp_layer(x).relu()
        x = self.hid_layer(x).relu()

        return self.out_layer(x)

    def __repr__(self):
        return f"MLP({self.inp_size}, {self.out_size}, {self.hid_size}, {self.n_hid}, {self.needs_bias})"

class Conv2d(Model):
    """
    The convolutional layer.
    """
    def __init__(self, inp_channels: int,
                 out_channels: int,
                 kernel_shape: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1):
        if not isinstance(inp_channels, int) or not isinstance(out_channels, int):
            raise TypeError("The number of input channels and output channels must be integers.")
        if inp_channels <= 0 or out_channels <= 0:
            raise ValueError("The number of input channels and output channels must be positive.")
        if not isinstance(kernel_shape, (int, tuple)):
            raise TypeError("The kernel shape must be an integer or a tuple.")

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape, kernel_shape)
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        random.seed(42)
        n_inp = out_channels * inp_channels * math.prod(kernel_shape)
        a = (6 / n_inp) ** 0.5
        self.weight = Parameter(Tensor([random.uniform(-a, a) for _ in range(n_inp)],
                                       shape = (out_channels, inp_channels, kernel_shape[0], kernel_shape[1])))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-3] != self.inp_channels:
            raise ValueError("The number of the input channels doesn't match.")
        return x.conv2d(self.weight, self.stride, self.padding, self.dilation)