from mytorch.tensor import *
from mytorch.operations import *
from mytorch.utils import *
from typing import Callable

from numpy import array # For faster matrix multiplication

def tensor_map(func) -> Callable:
    """
    A higher-level function which turns a float-to-float function into
    a tensor-to-tensor function.

    Args:
        func: A float-to-float function.

    Returns:
        A tensor-to-tensor function.
    """

    def _map(inp: Tensor, out_shape: tuple = None) -> Tensor:
        """
        The inner function which applies the given function on every element of the input tensor.
        It is also possible to specify the output shape to enable broadcasting.

        Args:
            inp (Tensor): The input tensor.
            out_shape (tuple): The shape of the output tensor after broadcasting.

        Returns:
            Tensor: The output tensor.

        """

        inp_storage, inp_shape, inp_stride, inp_offset = inp.core()
        inp_num = math.prod(inp_shape)

        if out_shape is None or (isinstance(out_shape, tuple) and inp_shape == out_shape): # Without broadcasting
            out_storage = [None] * inp_num
            for out_storage_idx in range(inp_num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, inp_shape) # Ensures contiguous layout for output
                inp_storage_idx = to_storage_idx(out_tensor_idx, inp_stride, inp_offset)
                out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])
            out = Tensor(out_storage, inp_shape)

        elif not isinstance(out_shape, tuple):
            raise TypeError(f"Expected tuple, got {type(out_shape)} instead.")
        elif len(inp_shape) > len(out_shape):
            raise ValueError("Invalid shapes for broadcasting.")
        else: # With broadcasting
            for i in range(-1, -len(inp_shape) - 1, -1):
                if inp_shape[i] != 1 and inp_shape[i] != out_shape[i]:
                    raise ValueError("Invalid shapes for broadcasting.")
                else:
                    pass
            out_num = math.prod(out_shape)
            out_storage = [None] * out_num
            for out_storage_idx in range(out_num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)  # Ensures contiguous layout for output
                inp_tensor_idx = from_broadcast_idx(out_tensor_idx, inp_shape)  # Broadcasting?????????
                inp_storage_idx = to_storage_idx(inp_tensor_idx, inp_stride, inp_offset)
                out_storage[out_storage_idx] = func(inp_storage[inp_storage_idx])
            out = Tensor(out_storage, out_shape)  # So we use the default contiguous stride for output

        return out

    return _map

def tensor_zip(func) -> Callable:
    """

    """
    def _zip(x: Tensor, y: Tensor) -> Tensor:
        """

        """

        x_storage, x_shape, x_stride, x_offset = x.core()
        y_storage, y_shape, y_stride, y_offset = y.core()

        if x_shape == y_shape: # Without broadcasting
            num = math.prod(x_shape)
            out_storage = [None] * num
            for out_storage_idx in range(num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, x_shape)

                x_storage_idx = to_storage_idx(out_tensor_idx, x_stride, x_offset)
                y_storage_idx = to_storage_idx(out_tensor_idx, y_stride, y_offset)

                out_storage[out_storage_idx] = func(x_storage[x_storage_idx], y_storage[y_storage_idx])
            out = Tensor(out_storage, x_shape)
        else: # With broadcasting
            if x.broadcastable(y) is False:
                raise ValueError(f"Tensors with shapes {x_shape} and {y_shape} are not broadcastable.")
            else:
                out_shape = x.broadcastable(y)
                out_num = math.prod(out_shape)
                out_storage = [None] * out_num

            for out_storage_idx in range(out_num):
                out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)

                x_tensor_idx = from_broadcast_idx(out_tensor_idx, x_shape)
                y_tensor_idx = from_broadcast_idx(out_tensor_idx, y_shape)

                x_storage_idx = to_storage_idx(x_tensor_idx, x_stride, x_offset)
                y_storage_idx = to_storage_idx(y_tensor_idx, y_stride, y_offset)

                out_storage[out_storage_idx] = func(x_storage[x_storage_idx], y_storage[y_storage_idx])
            out = Tensor(out_storage, out_shape)

        return out

    return _zip


def tensor_reduce(func) -> Callable:
    """

    """

    def _reduce(x: Tensor, dim: int, keepdim: bool = True) -> Union[Tensor, tuple]:
        """

        """

        x_storage, x_shape, x_stride, x_offset = x.core()

        out_shape = list(x_shape)
        out_shape[dim] = 1
        out_shape = tuple(out_shape)
        num = math.prod(out_shape)
        out_storage = [None] * num
        # When it returns the maximum value of a given dimension, we want to return
        # their indices within that dimension as well
        indices_storage = [None] * num

        for out_storage_idx in range(num):
            out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)
            for i in range(x_shape[dim]):
                x_tensor_idx = list(out_tensor_idx)
                x_tensor_idx[dim] = i
                x_tensor_idx = tuple(x_tensor_idx)
                x_storage_idx = to_storage_idx(x_tensor_idx, x_stride, x_offset)
                if out_storage[out_storage_idx] is None:
                    out_storage[out_storage_idx] = x_storage[x_storage_idx]
                else:
                    out_storage[out_storage_idx] = func(out_storage[out_storage_idx], x_storage[x_storage_idx])
                if func == maximum:
                    if out_storage[out_storage_idx] == x_storage[x_storage_idx]:
                        indices_storage[out_storage_idx] = i # type: ignore
                    else:
                        pass
                else:
                    pass
        if keepdim:
            if func == maximum:
                return (Tensor(out_storage, out_shape),
                        Tensor(indices_storage, out_shape))
            else:
                return Tensor(out_storage, out_shape)
        else:
            if len(out_shape) == 1:
                if func == maximum:
                    return (Tensor(out_storage, out_shape),
                            Tensor(indices_storage, out_shape))
                else:
                    return Tensor(out_storage, out_shape)
            else:
                if func == maximum:
                    return (Tensor(out_storage, out_shape[:dim] + out_shape[dim + 1:]),
                            Tensor(indices_storage, out_shape[:dim] + out_shape[dim + 1:]))
                else:
                    return Tensor(out_storage, out_shape[:dim] + out_shape[dim + 1:])

    return _reduce


def to_tensor(x: Union[float, int]) -> Tensor:
    """
    Convert some other types of data into our tensor, in order to allow computations between them.
    """
    if isinstance(x, (float, int)): # So far only float and int
        return Tensor([x])
    elif isinstance(x, Tensor):
        return x
    else:
        raise TypeError(f"{type(x)} is not supported for computations.")

def align(gradient: Tensor, shape: tuple) -> Tensor:
    """
    As we allow for broadcasting during computation, the shape of the output gradient might not
    match the shape of the respective tensor parameter, so this function is used to align
    their shape by sum over the broadcast dimensions of the output gradient.

    Args:
        gradient (Tensor): The gradient of the tensor parameter.
        shape (tuple): The shape of the tensor parameter which is also the target shape of our gradient.

    Returns:
        Tensor: The gradient whose shape is aligned.
    """

    grad_shape = gradient.shape
    if grad_shape != shape:
        for dim in range(-1, -len(shape) - 1, -1):
            if grad_shape[dim] != shape[dim]:
                gradient = gradient.sum(dim)
        if len(grad_shape) != len(shape):
            for _ in range(len(grad_shape) - len(shape)):
                gradient = gradient.sum(0, keepdim=False)
    return gradient

class Neg:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        neg_x = tensor_map(neg)(x)
        if x.history:
            neg_x.history = History(Neg, (), (x,))
        else:
            pass
        return neg_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        # We tend to use numpy array for tensors who don't need history
        return (-grad,)


class Exp:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        exp_x = tensor_map(exp)(x)
        if x.history:
            exp_x.history = History(Exp, exp_x.constant(), (x,)) # Tensors in cache should be constant
        else:
            pass
        return exp_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        exp_x = cache
        return (exp_x * grad,)

class Add:
    @staticmethod
    def forward(x: Tensor, y: Union[Tensor, float, int]) -> Tensor:
        y = to_tensor(y)
        add_x_y = tensor_zip(add)(x, y)
        if x.history or y.history:
            add_x_y.history = History(Add, (x.shape, y.shape), (x, y))
        else:
            pass
        return add_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x_shape, y_shape = cache
        return align(grad, x_shape), align(grad, y_shape)


class Sub:
    @staticmethod
    def forward(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
        x = to_tensor(x)
        y = to_tensor(y)
        sub_x_y = tensor_zip(sub)(x, y)
        if x.history or y.history:
            sub_x_y.history = History(Sub, (x.shape, y.shape), (x, y))
        else:
            pass
        return sub_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x_shape, y_shape = cache
        return align(grad, x_shape), align(-grad, y_shape)


class Mul:
    @staticmethod
    def forward(x: Tensor, y: Union[Tensor, float, int]) -> Tensor:
        y = to_tensor(y)
        mul_x_y = tensor_zip(mul)(x, y)
        if x.history or y.history:
            mul_x_y.history = History(Mul, (x.constant(), y.constant()), (x, y))
        else:
            pass
        return mul_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, y = cache
        return align(y * grad, x.shape), align(x * grad, y.shape)

class Div:
    @staticmethod
    def forward(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
        x = to_tensor(x)
        y = to_tensor(y)
        div_x_y = tensor_zip(div)(x, y)
        if x.history or y.history:
            cache = (x.constant(), y.constant())
            div_x_y.history = History(Div, cache, (x, y))
        else:
            pass
        return div_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, y = cache
        return align(grad / y, x.shape), align(-grad * x / (y ** 2), y.shape)


class Rcp:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        rcp_x = tensor_map(rcp)(x)
        if x.history:
            rcp_x.history = History(Rcp, rcp_x.constant(), (x,))
        else:
            pass
        return rcp_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        rcp_x = cache
        return (-grad * rcp_x ** 2,)


class Log:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        log_x = tensor_map(log)(x)
        if x.history:
            cache = x.constant()
            log_x.history = History(Log, cache, (x,))
        else:
            pass
        return log_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x = cache
        return (grad / x,)


class Sqrt:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        sqrt_x = tensor_map(sqrt)(x)
        if x.history:
            sqrt_x.history = History(Sqrt, sqrt_x.constant(), (x,))
        else:
            pass
        return sqrt_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        sqrt_x = cache
        return (grad / (2 * sqrt_x),)


class Max:
    @staticmethod
    def forward(x: Tensor, dim: int, keepdim: bool = True) -> tuple:
        max_x, indices = tensor_reduce(maximum)(x, dim, keepdim)
        if x.history:
            max_x.history = History(Max, (x.constant(), max_x.constant()), (x,))
        else:
            pass
        return max_x, indices

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, max_x = cache
        max_x *= Ones(grad.shape)

        return (grad * (max_x == x),)

class Relu:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        relu_x = tensor_map(relu)(x)
        if x.history:
            relu_x.history = History(Relu, x.constant(), (x,))
        else:
            pass
        return relu_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x = cache
        return (grad * (x > 0),)


class Softmax:
    """

    """
    @staticmethod
    def forward(x: Tensor, dim: int) -> Tensor:
        max_x = tensor_reduce(max)(x, dim) # Subtract the maximum to improve numerical stability
        exp_x = tensor_map(exp)(x - max_x)
        sum_exp_x = tensor_reduce(add)(exp_x, dim)
        softmax_x = exp_x / sum_exp_x
        if x.history:
            softmax_x.history = History(Softmax, (softmax_x.constant(), dim), (x,))
        else:
            pass
        return softmax_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        """

        """
        softmax_x, dim = cache

        return (softmax_x * (grad - tensor_reduce(add)(grad * softmax_x, dim)),)


class Sigmoid:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        sigmoid_x = tensor_map(sigmoid)(x)
        if x.history:
            sigmoid_x.history = History(Sigmoid, sigmoid_x.constant(), (x,))
        else:
            pass
        return sigmoid_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        sigmoid_x = cache
        return (sigmoid_x * (1 - sigmoid_x) * grad,)

class Pow:
    @staticmethod
    def forward(x: Union[Tensor, float, int], y: Union[Tensor, float, int]) -> Tensor:
        x = to_tensor(x)
        y = to_tensor(y)
        pow_x_y = tensor_zip(pow)(x, y)
        if x.history or y.history:
            cache = (x.constant(), y.constant())
            parents = (x, y)
            pow_x_y.history = History(Pow, cache, parents)
        else:
            pass
        return pow_x_y

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, y = cache
        return align(grad * y * x ** (y - 1), x.shape), align(x.log() * x ** y, y.shape)

class Reshape:
    @staticmethod
    def forward(x: Tensor, *shape) -> Tensor:
        if not x.is_contiguous():
            x.to_contiguous() # New storage
        else: # Same storage
            pass
        x_shape = x.shape
        if math.prod(x_shape) != math.prod(shape):
            raise ValueError(f"Can't reshape {x_shape} to {shape}.")
        else:
            reshape_x = Tensor(x.storage,
                               shape,
                               None,
                               x.offset)
        if x.history:
            reshape_x.history = History(Reshape, x_shape, (x,))
        else:
            pass
        return reshape_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x_shape = cache
        return (grad.reshape(*x_shape),)

class Permute:
    @staticmethod
    def forward(x: Tensor, *dims) -> Tensor:
        _, x_shape, x_stride, _ = x.core()
        if (any(d < 0 for d in dims) or
                len(x.shape) - 1 < max(dims) or
                len(x.shape) != len(dims) or
                len(dims) != len(set(dims))):
            raise ValueError("Invalid dimensions.")
        else: # Same storage
            permute_x = Tensor(x.storage,
                               tuple(x_shape[i] for i in dims),
                               tuple(x_stride[i] for i in dims),
                               x.offset)
        if x.history:
            permute_x.history = History(Permute, dims, (x,))
        else:
            pass
        return permute_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        dims = cache
        return (grad.permute(*dims),)

class Sum:
    @staticmethod
    def forward(x: Tensor, dim: int, keepdim: bool = True) -> Tensor:
        sum_x = tensor_reduce(add)(x, dim, keepdim)

        if x.history:
            sum_x.history = History(Sum, (x.shape, dim), (x,))
        else:
            pass
        return sum_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        out_shape, dim = cache
        if len(grad.shape) != len(out_shape):
            grad.shape = grad.shape[:dim] + (1,) + grad.shape[dim:]
            grad.stride = contiguous_stride(grad.shape)
        else:
            pass

        return (Ones(out_shape) * grad,)


class Prod:
    @staticmethod
    def forward(x: Tensor, dim: int, keepdim: bool = True) -> Tensor:
        prod_x = tensor_reduce(mul)(x, dim, keepdim)

        if x.history:
            prod_x.history = History(Prod, (x.constant(), prod_x.constant(), dim), (x,))
        else:
            pass
        return prod_x

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x, prod_x, dim = cache
        if len(grad.shape) != len(x.shape):
            grad.shape = grad.shape[:dim] + (1,) + grad.shape[dim:]
            grad.stride = contiguous_stride(grad.shape)

            prod_x.shape = prod_x.shape[:dim] + (1,) + prod_x.shape[dim:]
            prod_x.stride = contiguous_stride(prod_x.shape)
        else:
            pass

        return (grad * prod_x / x,)


class MatMul:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        """

        Args:

        Returns:

        """

        if not isinstance(y, Tensor):
            raise TypeError(f"Expected Tensor, got {type(y).__name__} instead.")

        x_storage, x_shape, x_stride, x_offset = x.core()
        y_storage, y_shape, y_stride, y_offset = y.core()
        x_remove = False # Check whether we need to remove imaginary dimension
        y_remove = False
        if len(x_shape) == 1: # Vector
            x_remove = True
            x_stride = x_shape + x_stride
            x_shape = (1,) + x_shape # Convert to a matrix with an extra imaginary dimension of size 1
        if len(y_shape) == 1:
            y_remove = True
            y_shape = y_shape + (1,)
            y_stride = (1,) + y_stride

        # The last two dimensions represent matrix, any extra dimensions ahead represent batch
        if x_shape[-1] != y_shape[-2]:
            raise ValueError(f"Shapes {x_shape[-2:]} and {y_shape[-2:]} are not compatible for matrix multiplication.")
        else:
            batch_shape = broadcastable(x_shape[:-2], y_shape[:-2]) # Batch shape can be broadcast
            if batch_shape is False:
                raise ValueError(f"Batch shapes are not broadcastable.")
            else:
                out_shape = batch_shape + (x_shape[-2], y_shape[-1])
                num = math.prod(out_shape)
                out_storage = [0] * num
                for out_storage_idx in range(num):
                    out_tensor_idx = to_tensor_idx(out_storage_idx, out_shape)
                    for i in range(x_shape[-1]):
                        x_tensor_idx = out_tensor_idx[:-1] + (i,)
                        x_tensor_idx = from_broadcast_idx(x_tensor_idx, x_shape)

                        y_tensor_idx = out_tensor_idx[:-2] + (i,) + out_tensor_idx[-1:]
                        y_tensor_idx = from_broadcast_idx(y_tensor_idx, y_shape)

                        x_storage_idx = to_storage_idx(x_tensor_idx, x_stride, x_offset)
                        y_storage_idx = to_storage_idx(y_tensor_idx, y_stride, y_offset)
                        out_storage[out_storage_idx] += x_storage[x_storage_idx] * y_storage[y_storage_idx]
                if x_remove:
                    out_shape = out_shape[:-2] + out_shape[-1:]
                if y_remove:
                    out_shape = out_shape[:-1]
                if len(out_shape) == 0:
                    out_shape = (1,) # Keep the shape (1, ) for scalar instead of ()
                out = Tensor(out_storage, out_shape)

                if x.history or y.history:
                    out.history = History(MatMul, (x.constant(), y.constant()), (x, y))
                else:
                    pass
                return out

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        """

        """

        x, y = cache
        x_shape = x.shape # Remember the original shape to adjust the shape of the output gradient,
        # as they must align with each other
        y_shape = y.shape
        x_remove = False
        y_remove = False

        if len(x.shape) == 1 and len(y.shape) != 1:
            x_remove = True

            x.stride = x.shape + x.stride
            x.shape = (1,) + x.shape

            grad.stride = grad.stride[:-1] + (grad.stride[-1],) + grad.stride[-1:]
            grad.shape = grad.shape[:-1] + (1,) + grad.shape[-1:]
            grad.stride = contiguous_stride(grad.shape)

        if len(y.shape) == 1 and len(x.shape) != 1:
            y_remove = True

            y.shape = y.shape + (1,)
            y.stride = (1,) + y.stride

            grad.shape = grad.shape + (1,)
            grad.stride = grad.stride + (1,)

        if len(y.shape) == len(x.shape) == 1:
            x_remove = True
            y_remove = True

            x.stride = x.shape + x.stride
            x.shape = (1,) + x.shape

            y.shape = y.shape + (1,)
            y.stride = (1,) + y.stride

            grad.shape = (1, 1)
            grad.stride = (1, 1)

        def transpose(t: Tensor):
            t_shape = list(t.shape)
            t_stride = list(t.stride)
            t_shape[-1], t_shape[-2] = t_shape[-2], t_shape[-1]
            t_stride[-1], t_stride[-2] = t_stride[-2], t_stride[-1]
            t.shape = tuple(t_shape)
            t.stride = tuple(t_stride)

        transpose(x)
        transpose(y)

        x_grad = grad @ y
        y_grad = x @ grad

        if x_remove:
            x_grad = Tensor(x_grad.storage, x_grad.shape[:-2] + x_grad.shape[-1:])
        if y_remove:
            y_grad = Tensor(y_grad.storage, y_grad.shape[:-1])

        return align(x_grad, x_shape), align(y_grad, y_shape)



class FastMatMul:
    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        if not isinstance(y, Tensor):
            raise TypeError(f"Expected Tensor, got {type(y).__name__} instead.")
        if not x.is_contiguous(): # Only storage in contiguous layout can use numpy reshape
            x.to_contiguous()
        if not y.is_contiguous():
            y.to_contiguous()

        x_storage, x_shape, _, x_offset = x.core()
        y_storage, y_shape, _, y_offset = y.core()

        x_ndarray = array(x_storage[x_offset:]).reshape(x_shape)
        y_ndarray = array(y_storage[y_offset:]).reshape(y_shape)

        out_ndarray = x_ndarray @ y_ndarray
        out = Tensor(out_ndarray.flatten().tolist(), out_ndarray.shape)

        if x.history or y.history:
            out.history = History(FastMatMul, (x_ndarray, y_ndarray), (x, y))

        return out

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x_ndarray, y_ndarray = cache
        x_shape = x_ndarray.shape
        y_shape = y_ndarray.shape

        if not grad.is_contiguous():
            grad.to_contiguous()

        grad_storage, grad_shape, _, grad_offset = grad.core()
        grad_ndarray = array(grad_storage[grad_offset:]).reshape(grad_shape)

        y_axes = list(range(y_ndarray.ndim))
        y_axes[-2], y_axes[-1] = y_axes[-1], y_axes[-2]
        x_grad_ndarray = grad_ndarray @ y_ndarray.transpose(y_axes)

        x_axes = list(range(x_ndarray.ndim))
        x_axes[-2], x_axes[-1] = x_axes[-1], x_axes[-2]
        y_grad_ndarray = x_ndarray.transpose(x_axes) @ grad_ndarray

        x_grad = Tensor(x_grad_ndarray.flatten().tolist(), x_grad_ndarray.shape)
        y_grad = Tensor(y_grad_ndarray.flatten().tolist(), y_grad_ndarray.shape)

        return align(x_grad, x_shape), align(y_grad, y_shape)


class MyConv2d:
    """

    """
    @staticmethod
    def forward(x: Tensor,
                k: Tensor,
                stride: Union[int, tuple] = 1,
                padding: Union[int, tuple] = 0,
                dilation: Union[int, tuple] = 1) -> Tensor:
        """

        Args:
            x:
            k:
            stride:
            padding:
            dilation:

        Returns:

        """
        if not isinstance(k, Tensor):
            raise TypeError(f"Expected Tensor, got {type(k).__name__} instead.")
        x_storage, x_shape, x_stride, x_offset = x.core()
        k_storage, k_shape, k_stride, k_offset = k.core()

        batch_size, inp_channels, h, w = x_shape
        out_channel, inp_channels_, kh, kw = k_shape

        if inp_channels != inp_channels_:
            raise ValueError(f"Mismatch in the number of input channels, got {inp_channels} and {inp_channels_}.")
        if not isinstance(stride, (int, tuple)):
            raise TypeError("The stride must be an integer or a tuple.")
        if not isinstance(padding, (int, tuple)):
            raise TypeError("The padding must be an integer or a tuple.")
        if not isinstance(dilation, (int, tuple)):
            raise TypeError("The dilation must be an integer or a tuple.")

        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if padding[0] < 0 or padding[1] < 0:
            raise ValueError(f"Padding must be non-negative, got {padding} instead.")
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError(f"Stride must be positive, got {stride} instead.")
        if dilation[0] <= 0 or dilation[1] <= 0:
            raise ValueError(f"Dilation must be positive, got {dilation} instead.")
        if dilation[0] * (kh - 1) + 1 > h + 2 * padding[0] or dilation[1] * (kw - 1) + 1 > w + 2 * padding[1]:
            raise ValueError("The (dilated) kernel shape is larger than the (padded) input shape")

        if padding != (0, 0):
            x_padded_shape = (batch_size, inp_channels, h + padding[0] * 2, w + padding[1] * 2)
            num = math.prod(x_padded_shape)
            x_padded_storage = [0] * num
            for x_padded_storage_idx in range(num):
                x_padded_tensor_idx = to_tensor_idx(x_padded_storage_idx, x_padded_shape)
                if (padding[1] <= x_padded_tensor_idx[-1] <= w - 1 + padding[1] and
                        padding[0] <= x_padded_tensor_idx[-2] <= h - 1 + padding[0]):
                    x_tensor_idx = list(x_padded_tensor_idx)
                    x_tensor_idx[-1] -= padding[1]
                    x_tensor_idx[-2] -= padding[0]
                    x_tensor_idx = tuple(x_tensor_idx)
                    x_storage_idx = to_storage_idx(x_tensor_idx, x_stride, x_offset)
                    x_padded_storage[x_padded_storage_idx] = x_storage[x_storage_idx]
            x_padded = Tensor(x_padded_storage, x_padded_shape)
        else:
            x_padded = x

        if dilation != (1, 1):
            k_dilated_shape = (out_channel, inp_channels, dilation[0] * (kh - 1) + 1, dilation[1] * (kw - 1) + 1)
            num = math.prod(k_dilated_shape)
            k_dilated_storage = [0] * num
            for k_dilated_storage_idx in range(num):
                k_dilated_tensor_idx = to_tensor_idx(k_dilated_storage_idx, k_dilated_shape)
                if k_dilated_tensor_idx[-1] % dilation[-1] == 0 and k_dilated_tensor_idx[-2] % dilation[-2] == 0:
                    k_tensor_idx = list(k_dilated_tensor_idx)
                    k_tensor_idx[-1] //= dilation[-1] # Use '//' because index has to be integer
                    k_tensor_idx[-2] //= dilation[-2]
                    k_tensor_idx = tuple(k_tensor_idx)
                    k_storage_idx = to_storage_idx(k_tensor_idx, k_stride, k_offset)
                    k_dilated_storage[k_dilated_storage_idx] = k_storage[k_storage_idx]
            k_dilated = Tensor(k_dilated_storage, k_dilated_shape)
        else:
            k_dilated = k

        x_padded_storage, _, x_padded_stride, x_padded_offset = x_padded.core()
        _, _, dkh, dkw = k_dilated.shape
        # The shape of the output
        out_h = ((h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0]) + 1
        out_w = ((w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1]) + 1

        # Here we use Im2Col algorithm to change convolution into matrix multiplication.
        # So we first have to construct the new input and kernel
        x_new_shape = (batch_size, out_h, out_w, inp_channels * dkh * dkw) # The shape of the 'flattened' input
        num = math.prod(x_new_shape)
        x_new_storage = [0] * num
        for x_new_storage_idx in range(num):
            x_new_tensor_idx = to_tensor_idx(x_new_storage_idx, x_new_shape)
            part_idx = list(to_tensor_idx(x_new_tensor_idx[-1], (inp_channels, dkh, dkw)))
            part_idx[1] += x_new_tensor_idx[1] * stride[0]
            part_idx[2] += x_new_tensor_idx[2] * stride[1]
            part_idx = tuple(part_idx)
            x_padded_tensor_idx = (x_new_tensor_idx[0],) + part_idx
            x_padded_storage_idx = to_storage_idx(x_padded_tensor_idx, x_padded_stride, x_padded_offset)
            x_new_storage[x_new_storage_idx] = x_padded_storage[x_padded_storage_idx]
        x_new = Tensor(x_new_storage, x_new_shape)
        # Also need to reshape the dilated kernel
        k_new = k_dilated.reshape(out_channel, inp_channels * dkh * dkw).permute(1, 0)
        out = x_new @ k_new # Then we can apply matrix multiplication between the new input and kernel
        out = out.permute(0, 3, 1, 2) # Change its shape back to how it should be
        if x.history or k.history:
            out.history = History(MyConv2d, (x_padded.constant(), k_dilated.constant(),
                                           stride, padding, dilation), (x, k))

        return out

    @staticmethod
    def backward(cache, grad: Tensor) -> tuple:
        x_padded, k_dilated, stride, padding, dilation = cache
        x_padded_storage, x_padded_shape, x_padded_stride, x_padded_offset = x_padded.core()
        k_dilated_storage, k_dilated_shape, k_dilated_stride, k_dilated_offset = k_dilated.core()
        # Sometimes our kernel can't reach the end of the input, in order to generate the correct gradient,
        # we need to record the number of rows and columns that are not used in the forward pass
        unused = [0, 0]
        unused[0] = (x_padded_shape[-2] - k_dilated_shape[-2]) % stride[0] # Number of unused rows at the bottom
        unused[1] = (x_padded_shape[-1] - k_dilated_shape[-1]) % stride[1] # Number of unused columns on the right

        # Change the padded input accordingly by removing unused rows and columns
        if unused[0] != 0 or unused[1] != 0:
            x_padded_used_shape = (x_padded_shape[0],
                                   x_padded_shape[1],
                                   x_padded_shape[2] - unused[0],
                                   x_padded_shape[3] - unused[1])
            num = math.prod(x_padded_used_shape)
            x_padded_used_storage = [0] * num
            for x_padded_used_storage_idx in range(num):
                x_padded_used_tensor_idx = to_tensor_idx(x_padded_used_storage_idx, x_padded_used_shape)
                if (x_padded_used_tensor_idx[-1] <= x_padded_shape[-1] - 1 - unused[1] and
                        x_padded_used_tensor_idx[-2] <= x_padded_shape[-2] - 1 - unused[0]):
                    x_padded_storage_idx = to_storage_idx(x_padded_used_tensor_idx, x_padded_stride, x_padded_offset)
                    x_padded_used_storage[x_padded_used_storage_idx] = x_padded_storage[x_padded_storage_idx]
            x_padded_used = Tensor(x_padded_used_storage, x_padded_used_shape)

            # Change the dilated kernel accordingly by padding zeros
            k_dilated_used_shape = (k_dilated_shape[0],
                                   k_dilated_shape[1],
                                   k_dilated_shape[2] + unused[0],
                                   k_dilated_shape[3] + unused[1])
            num = math.prod(k_dilated_used_shape)
            k_dilated_used_storage = [0] * num
            for k_dilated_used_storage_idx in range(num):
                k_dilated_used_tensor_idx = to_tensor_idx(k_dilated_used_storage_idx, k_dilated_used_shape)
                if (k_dilated_used_tensor_idx[-1] <= k_dilated_shape[-1] - 1 and
                        k_dilated_used_tensor_idx[-2] <= k_dilated_shape[-2] - 1):
                    k_dilated_storage_idx = to_storage_idx(k_dilated_used_tensor_idx, k_dilated_stride, k_dilated_offset)
                    k_dilated_used_storage[k_dilated_used_storage_idx] = k_dilated_storage[k_dilated_storage_idx]
            k_dilated_used = Tensor(k_dilated_used_storage, k_dilated_used_shape)
        else:
            x_padded_used = x_padded
            k_dilated_used = k_dilated

        k_grad = MyConv2d.forward(x_padded_used.permute(1, 0, 2, 3),
                                grad.permute(1, 0, 2, 3),
                                stride=dilation,
                                padding=0,
                                dilation=stride).permute(1, 0, 2, 3)

        # When computing the gradient w.r.t. the input, we first need to rotate the gradient
        grad_storage, grad_shape, grad_stride, grad_offset = grad.core()
        grad_r_shape = grad_shape
        num = math.prod(grad_r_shape)
        grad_r_storage = [0] * num
        for grad_r_storage_idx in range(num):
            grad_r_tensor_idx = to_tensor_idx(grad_r_storage_idx, grad_r_shape)
            grad_tensor_idx = list(grad_r_tensor_idx)
            grad_tensor_idx[-1] = grad_shape[-1] - 1 - grad_r_tensor_idx[-1]
            grad_tensor_idx[-2] = grad_shape[-2] - 1 - grad_r_tensor_idx[-2]
            grad_tensor_idx = tuple(grad_tensor_idx)
            grad_storage_idx = to_storage_idx(grad_tensor_idx, grad_stride, grad_offset)
            grad_r_storage[grad_r_storage_idx] = grad_storage[grad_storage_idx]
        grad_r = Tensor(grad_r_storage, grad_r_shape)

        x_grad = MyConv2d.forward(k_dilated_used.permute(1, 0, 2, 3),
                                grad_r,
                                stride=1,
                                padding=(stride[0] * (grad_shape[-2] - 1) - padding[-2],
                                         stride[1] * (grad_shape[-1] - 1) - padding[-1]),
                                dilation=stride).permute(1, 0, 2, 3)

        return x_grad, k_grad


Tensor.__neg__ = Neg.forward
Tensor.__add__ = Add.forward
Tensor.__radd__ = Add.forward
Tensor.__sub__ = Sub.forward
Tensor.__rsub__ = lambda x, y: Sub.forward(y, x)
Tensor.__mul__ = Mul.forward
Tensor.__rmul__ = Mul.forward
Tensor.__truediv__ = Div.forward
Tensor.__rtruediv__ = lambda x, y: Div.forward(y, x)
Tensor.__eq__ = lambda x, y: tensor_zip(eq)(x, to_tensor(y))
Tensor.__gt__ = lambda x, y: tensor_zip(gt)(x, to_tensor(y))
Tensor.__lt__ = lambda x, y: tensor_zip(lt)(x, to_tensor(y))
Tensor.__ge__ = lambda x, y: tensor_zip(ge)(x, to_tensor(y))
Tensor.__le__ = lambda x, y: tensor_zip(le)(x, to_tensor(y))
Tensor.__pow__ = Pow.forward
Tensor.__rpow__ = lambda x, y: Pow.forward(y, x)
Tensor.__matmul__ = FastMatMul.forward
Tensor.conv2d = MyConv2d.forward
Tensor.max = Max.forward
Tensor.sum = Sum.forward
Tensor.prod = Prod.forward
Tensor.log = Log.forward
Tensor.sqrt = Sqrt.forward
Tensor.softmax = Softmax.forward
Tensor.sigmoid = Sigmoid.forward
Tensor.exp = Exp.forward
Tensor.relu = Relu.forward
Tensor.rcp = Rcp.forward
Tensor.reshape = Reshape.forward
Tensor.permute = Permute.forward










