import math
from typing import Union


def to_storage_idx(tensor_idx: tuple, stride: tuple, offset: int) -> int:
    """
    Given the indices of an element within a tensor, the function returns the index of
    the element within the storage of the tensor.

    Args:
        tensor_idx (tuple): The index of the element within the tensor.
        stride (tuple): The stride of the tensor.
        offset (int): The offset of the tensor.

    Returns:
        int: The index of the element within the storage of the tensor.

    """
    storage_idx = 0

    for i in range(len(tensor_idx)):
        storage_idx += tensor_idx[i] * stride[i]

    return storage_idx + offset


def to_tensor_idx(storage_idx: int, shape: tuple) -> tuple:
    """
    Given the indices of an element within the storage of a tensor, the function returns the index of
    the element within the tensor based on contiguous layout, regardless of the stride.

    Args:
        storage_idx (int): The index of the element within the storage of the tensor.
        shape (tuple): The shape of the tensor.

    Returns:
        tuple: The index of the element within the tensor based on contiguous layout.

    """
    if storage_idx + 1 > math.prod(shape):
        raise ValueError("Index out of bounds.")

    tensor_idx = [0] * len(shape)

    for i in range(len(shape) - 1, -1, -1):
        tensor_idx[i] = storage_idx % shape[i]
        storage_idx //= shape[i]

    return tuple(tensor_idx)


def from_broadcast_idx(broadcast_idx: tuple, shape: tuple) -> tuple:
    """
    Given the index of an element within a broadcast tensor, this function returns its index within
    the original tensor before broadcasting, which means no new copies need to be made to address the
    distinction in shape, less memory will be used, more efficient.

    Args:
        broadcast_idx: The index of the element within the broadcast tensor.
        shape: The shape of the original tensor before broadcasting.

    Returns:
        tuple: The index of the element within the original tensor before broadcasting.

    """
    index = [0] * len(shape)
    for i in range(-1, -len(shape) - 1, -1):
        if shape[i] == 1:
            index[i] = 0
        else:
            index[i] = broadcast_idx[i]
    return tuple(index)


def layout(shape):
    """
    Create a nested list of zeros with the given shape.
    """
    if len(shape) == 0:
        return None
    else:
        pass
    return [layout(shape[1:]) for _ in range(shape[0])]


def assign(lst: list, pos: tuple, value: float):
    """
    Assign a new value to a nested list given a tuple.
    """
    target = lst
    for i in pos[:-1]:
        target = target[i]
    target[pos[-1]] = value


def broadcastable(shape1: tuple, shape2: tuple) -> Union[bool, tuple]:
    """
    According to the broadcasting rules, we check whether two shapes are broadcastable,
    if they aren't, return False, if they are, return the broadcast shape.
    """

    if len(shape1) > len(shape2):
        broadcast_shape = list(shape1)
    else:
        broadcast_shape = list(shape2)
    for i in range(-1, -min(len(shape1), len(shape2)) - 1, -1):
        if shape1[i] != shape2[i]:
            if shape1[i] != 1:
                if shape2[i] != 1:
                    return False
                else:
                    broadcast_shape[i] = shape1[i]
            else:
                broadcast_shape[i] = shape2[i]
        else:
            broadcast_shape[i] = shape1[i]
    return tuple(broadcast_shape)


def contiguous_stride(shape: tuple):
    """
    Return the contiguous stride of the tensor.
    """

    stride = [1] * len(shape)
    for i, s in enumerate(tuple(reversed(shape[1:]))):
        stride[i + 1] = stride[i] * s
    return tuple(reversed(stride))