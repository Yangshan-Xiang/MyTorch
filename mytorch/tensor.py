from typing import Union
from mytorch.utils import *

class History:
    """
    Store the computation history of a tensor parameter which will be used in the backward pass.

    Attributes:
        function: The function which is used.
        cache: Stores some useful values computed during forward pass to save computation.
        parents (tuple): The input values for the function.

    """
    def __init__(self, function = None, cache = None, parents: tuple = None):
        self.function = function
        self.cache = cache
        self.parents = parents

    def __repr__(self):
        if self.parents is None:
            return f"History()"
        else:
            return f"History(function={self.function}, parents={self.parents})"

class Tensor:
    """
    Store the four core features of a tensor, and its history and gradient if it is a tensor parameter.

    Attributes:
        storage (list): A 1-dimensional list which stores the value of the elements of the tensor.
        shape (tuple): The shape of the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.
        history (History): Stores the computation history of the tensor parameter.
    """

    def __init__(self,
                 storage: list,
                 shape: tuple = None,
                 stride: tuple = None,
                 offset: int = 0,
                 history: History = None):

        if not isinstance(storage, list):
            raise TypeError(f"Storage must be a list, got {type(storage).__name__} instead.")
        else:
            self.storage = storage

        if shape is None:
            self.shape = (len(storage),) # Default shape is a 1-d vector
        elif not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape).__name__} instead.")
        elif len(self.storage) < math.prod(shape):
            # As we can just use a part of the storage, offset is needed
            raise ValueError("Tensor of given shape requires more elements than its storage has.")
        elif not all(isinstance(s, int) for s in shape):
            raise TypeError("Shape can only contain integers.")
        elif not all(s > 0 for s in shape):
            raise ValueError("shape can only contain positive integers.")
        else:
            self.shape = shape

        if stride is None:
            # If stride is not given, we will use the corresponding contiguous stride
            self.stride = self.contiguous_stride()
        elif not isinstance(stride, tuple):
            raise TypeError(f"Stride must be a tuple, got {type(stride).__name__} instead.")
        elif len(stride) != len(self.shape):
            raise ValueError("Invalid stride for shape.")
        elif not all(isinstance(s, int) for s in stride):
            raise TypeError("Stride can only contain integers.")
        elif not all(s > 0 for s in stride):
            raise ValueError("Stride can only contain positive integers.")
        elif (sum([math.prod(l) for l in list(zip([s - 1 for s in self.shape], [*stride]))])
              > math.prod(self.shape) - 1):
            raise ValueError("Invalid stride for shape.")
        else:
            self.stride = stride

        if not isinstance(offset, int):
            raise TypeError(f"Offset must be a int, got {type(offset).__name__} instead.")
        elif offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        elif len(self.storage) < math.prod(self.shape) + offset:
            raise ValueError(f"Given storage and shape, offset can't be bigger "
                             f"than {len(self.storage) - math.prod(self.shape)}.")
        else:
            self.offset = offset

        self.history = history
        self.gradient = None

    def core(self):
        return self.storage, self.shape, self.stride, self.offset

    def is_constant(self):
        """
        Check if the tensor is constant, if it is, then we don't need to compute the gradient w.r.t. it.
        Be aware that we didn't use 'requires_grad' here like what PyTorch did, so make sure that you have
        initialized the history of the tensor with 'History()' when you want the gradient w.r.t. it to be
        computed.
        """

        return self.history is None

    def is_parameter(self):
        return self.history is not None

    def to_contiguous(self) -> None:
        """
        Convert the tensor to a contiguous layout, be aware that it will also remove unused
        elements within the storage.
        """

        if self.is_contiguous():
            raise AssertionError("Tensor is already contiguous.")
        else:
            self_storage, self_shape, self_stride, self_offset = self.core()
            num = math.prod(self_shape)
            out_storage = [None] * num
            for out_storage_idx in range(num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, self_shape)
                self_storage_idx = to_storage_idx(out_tensor_idx, self_stride, self_offset)
                out_storage[out_storage_idx] = self_storage[self_storage_idx]
            self.storage = out_storage
            self.stride = self.contiguous_stride()
            self.offset = 0

    def constant(self):
        """
        Return a constant copy of a tensor parameter by simply removing its history.
        """

        return Tensor(self.storage, self.shape, self.stride, self.offset)

    def update(self, other: 'Tensor'):
        """
        Update the four core features of a tensor parameter given another tensor.
        """

        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor, got {type(other).__name__} instead.")
        else:
            self.storage = other.storage
            self.shape = other.shape
            self.stride = other.stride
            self.offset = other.offset

    def is_leaf(self):
        """
        Check if the tensor parameter is a leaf parameter, as we only need to store the gradients
        w.r.t. leaf parameters.
        """

        return self.history is not None and self.history.parents is None

    def is_contiguous(self):
        """
        Check if the tensor's stride is contiguous.
        """
        return self.stride == self.contiguous_stride()

    def contiguous_stride(self):
        """
        Return the contiguous stride of the tensor.
        """
        return contiguous_stride(self.shape)

    def broadcastable(self, other: 'Tensor') -> Union[bool, tuple]:
        """
        According to the broadcasting rules, we check whether the two tensors are broadcastable,
        if they aren't, return False, if they are, return the shape of the broadcast tensor.

        Args:
            self (Tensor): The tensor itself.
            other (Tensor): The other tensor.

        Returns:
            Union[bool, tuple]: False if the inputs are not broadcastable, the shape of the broadcast tensor otherwise.
        """

        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor, got {type(other).__name__} instead.")
        else:
            return broadcastable(self.shape, other.shape)

    def __getitem__(self):
        pass
    def __setitem__(self):
        pass

    def topological_sort(self) -> list:
        """
        Find the topological ordering of tensor parameters. Here we apply depth-first search
        with reverse post-ordering to linearly order the parameters. The result might
        not be unique, it ensures that a parent parameter will only be visited
        after all of its children have been visited to avoid redundant computations.
        """

        visited = [] # Avoid revisit
        ordered = []

        def dfs(x):
            if x.history: # We don't need constant tensors here
                if id(x) not in visited: # Use id() to identify different parameters
                    visited.append(id(x))

                    if x.history.parents:
                        for parent in x.history.parents:
                            dfs(parent) # Recursion
                        ordered.append(x)
                    else:
                        ordered.append(x)
                else:
                    pass
            else:
                pass

        dfs(self)
        ordered.reverse() # Reverse post-ordering
        return ordered

    def chain_rule(self, grad: 'Tensor') -> list:
        """
        Apply the chain rule to compute and assign the derivatives of the child with respect to its parents.
        """

        h = self.history
        return list(zip(h.parents, h.function.backward(h.cache, grad))) # Pair the gradients with its parents

    def backward(self) -> None:
        """
        Apply the chain rule to compute and save the gradients w.r.t. the leaf tensor parameters.
        The saved gradients will then be used by optimizers to update themselves.
        """

        ordered = self.topological_sort() # No constant tensors already
        catalog = {id(x): Tensor([0]) for x in ordered} # Record computed gradients
        # Initialize the starting gradient as an all-ones tensor
        catalog[id(self)] = Tensor([1] * math.prod(self.shape), self.shape)

        for param in ordered:
            if param.history.parents: # Not a leaf parameter
                for parent, grad in param.chain_rule(catalog[id(param)]):
                    if parent.history:
                        catalog[id(parent)] += grad # Accumulate the gradients according to the chain rule
                    else:
                        pass
            else: # A leaf parameter
                if not param.gradient:
                    param.gradient = Tensor([0])
                else:
                    pass
                param.gradient += catalog[id(param)] # Enable it to accumulate derivatives,
                # useful in certain cases like multitask learning.
    def tolist(self):
        """
        Return the tensor as a nested list.
        """
        out = layout(self.shape)
        self_storage, self_shape, self_stride, self_offset = self.core()
        num = math.prod(self.shape)
        for out_storage_idx in range(num):
            out_tensor_idx = to_tensor_idx(out_storage_idx, self_shape)
            self_storage_idx = to_storage_idx(out_tensor_idx, self_stride, self_offset)
            assign(out, out_tensor_idx, round(self_storage[self_storage_idx], 4))

        return out

    def __repr__(self):
        if self.history is None:
            return f"Tensor({self.tolist()})"
        else:
            return f"Tensor({self.tolist()}, history={self.history})"


    # To avoid circular import, following methods are all defined in arithmetics.py
    def __add__(self, other):
        pass
    def __radd__(self, other):
        pass
    def __sub__(self, other):
        pass
    def __rsub__(self, other):
        pass
    def __mul__(self, other):
        pass
    def __rmul__(self, other):
        pass
    def __truediv__(self, other):
        pass
    def __rtruediv__(self, other):
        pass
    def __eq__(self, other):
        pass
    def __gt__(self, other):
        pass
    def __lt__(self, other):
        pass
    def __neg__(self):
        pass
    def __pow__(self):
        pass
    def __matmul__(self, other):
        pass
    def conv2d(self, other,
               stride: Union[int, tuple] = 1,
               padding: Union[int, tuple] = 0,
               dilation: Union[int, tuple] = 1):
        pass
    def max(self, dim: int, keepdim: bool = True):
        pass
    def softmax(self, dim: int):
        pass
    def sigmoid(self):
        pass
    def log(self):
        pass
    def sqrt(self):
        pass
    def exp(self):
        pass
    def relu(self):
        pass
    def rcp(self):
        pass
    def reshape(self, *shape):
        pass
    def permute(self, *dims):
        pass
    def sum(self, dim: int, keepdim: bool = True):
        pass
    def prod(self, dim: int, keepdim: bool = True):
        pass

class Ones(Tensor):
    """
    All-one tensor
    """
    def __init__(self, shape: tuple, history: History = None):
        storage = [1] * math.prod(shape)
        super().__init__(storage, shape, history = history)

class Zeros(Tensor):
    """
    All-zero tensor
    """
    def __init__(self, shape: tuple, history: History = None):
        storage = [0] * math.prod(shape)
        super().__init__(storage, shape, history = history)










