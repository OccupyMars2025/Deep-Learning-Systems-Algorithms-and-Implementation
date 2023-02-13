"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Union
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp

from .autograd import TensorTuple, TensorTupleOp

import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value: TensorTuple, index: int):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a: NDArray):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a: Tensor, b: Tensor):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a: Tensor, b: Tensor):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        # return a tuple
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raises a tensor to an (integer) power."""
    #TODO:why does scalar have to be an integer ???
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION
        x = node.inputs
        return out_grad * (power_scalar(x[0], self.scalar - 1) * self.scalar)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION
        return a / b 

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION
        _, b = node.inputs
        return divide(out_grad, b), out_grad * (divide(negate(node), b))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar 

    def gradient(self, out_grad, node):
        return divide_scalar(out_grad, self.scalar)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    """
    Interchange two axes of an array.
    If "axes" is None, then interchange the last two axes.
    """
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if axes is not None:
            assert len(axes) == 2

    def compute(self, a):
        # # wrong!
        # return array_api.transpose(a, self.axes)
        if self.axes is not None:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, -1, -2)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs
        return reshape(out_grad, a[0].shape)


def reshape(a: Tensor, shape: List[int]):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        #TODO 2023-2-6 14:00 Can this implementation be simplified ???
        original_shape = node.inputs[0].shape
        broadcast_shape = self.shape
        axes_to_sum_over = []
        for i in range(-1, -len(broadcast_shape) - 1, -1):
            if i >= -len(original_shape):
                if 1 == original_shape[i]:
                    axes_to_sum_over.append(i)
            else:
                axes_to_sum_over.append(i)
        # np.broadcast_to doesn't accept a list as the "axis" argument,
        # because now I use numpy as the backend, so I have to transform the list to a tuple
        axes_to_sum_over = tuple(axes_to_sum_over)
        
        temp = summation(out_grad, axes=axes_to_sum_over, keep_axes=False)
        return reshape(temp, shape=original_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None, keep_axes: bool = False):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes, )
        else:
            self.axes = axes
        self.keep_axes = keep_axes

    def compute(self, a: NDArray):
        return array_api.sum(a, axis=self.axes, keepdims=self.keep_axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        original_shape = node.inputs[0].shape
        if self.axes and not self.keep_axes:
            shape_that_keeps_axes = list(original_shape)
            for i in self.axes:
                shape_that_keeps_axes[i] = 1
            temp = reshape(out_grad, shape_that_keeps_axes)
            return broadcast_to(temp, original_shape)
        else:
            return broadcast_to(out_grad, original_shape)


def summation(a: Tensor, axes: Optional[Union[tuple, int]] = None, keep_axes: bool=False):
    return Summation(axes, keep_axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return array_api.matmul(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        x1, x2 = node.inputs
        gradient_wrt_x1 = matmul(out_grad, transpose(x2))
        gradient_wrt_x2 = matmul(transpose(x1), out_grad)
        # TODO: According to my intuition, it needs to use summation. But it needs some math to clarify it.
        # Caution !!!
        # If the shape of the gradient w.r.t x is (6, 6, 5, 4), but x.shape is (5, 4)
        # then you should use sum(gradient, axes=(-3, -4)) to get the correct gradient.
        if gradient_wrt_x1.shape != x1.shape:
            gradient_wrt_x1 = summation(gradient_wrt_x1, tuple(range(-len(x1.shape) -1, -len(gradient_wrt_x1.shape) - 1, -1)))
        if gradient_wrt_x2.shape != x2.shape:
            gradient_wrt_x2 = summation(gradient_wrt_x2, tuple(range(-len(x2.shape) -1, -len(gradient_wrt_x2.shape) - 1, -1)))        
        return gradient_wrt_x1, gradient_wrt_x2


def matmul(a: Tensor, b: Tensor):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        return array_api.negative(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return negate(out_grad)


def negate(a: Tensor):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return divide(out_grad, node.inputs[0])


def log(a: Tensor):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return multiply(out_grad, node)


def exp(a: Tensor):
    return Exp()(a)


class ReLU(TensorOp):
    """
    TODO: 2023/2/8 16:40 Does this in-place modification work ???
    refer to tests/my_playground/relu.py
    """
    def compute(self, a: NDArray):
        # # It seems that the in-place version cannot pass "gradient_check" in tests\test_autograd_hw.py
        # # This is an in-place modification for np.ndarray
        # a[a < 0] = 0
        # return a 

        return array_api.where(a < 0, 0, a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        "in this one case it's acceptable to access the .realize_cached_data() 
        call on the output tensor, since the ReLU function is not twice differentiable anyway" ???
        TODO: The above is quoted from the hw1 notes, but what does it mean ????
        """
        # Implementation 1: This implementation seems to be wrong !!!
        # node_np = node.realize_cached_data()
        # node_np[node_np > 0] = 1
        # return out_grad * Tensor(node_np)

        # Implementation 2: Caution: node tensor will not be one of the inputs of the gradient tensor 
        # TODO: What will be the effect ???
        node_np = node.realize_cached_data()
        return out_grad * Tensor(array_api.where(node_np > 0, 1, 0))


def relu(a: Tensor):
    return ReLU()(a)


class LogSumExp(TensorOp):
    """
    axes - Tuple of axes to sum and take the maximum element over

    Applies a "numerically stable" log-sum-exp function to the input 
    by subtracting off the maximum elements.

    This uses the same conventions as needle.ops.Summation()
    """
    def __init__(self, axes: Optional[Union[tuple, int]] = None, keep_axes: bool = False):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes, )
        else:
            self.axes = axes
        self.keep_axes = keep_axes

    def compute(self, Z: NDArray):
        # "compute" use numpy to calculate, so they can broadcast implicitly
        max_elements_along_axes = array_api.max(Z, axis=self.axes, keepdims=True)
        result = array_api.exp(Z - max_elements_along_axes)
        result = array_api.sum(result, axis=self.axes, keepdims=self.keep_axes)
        if self.keep_axes:
            return array_api.log(result) + max_elements_along_axes
        else:
            # You CANNOT use array_api.squeeze(max_elements_along_axes).
            # Because there may exist an axis i such that Z.shape[i] = 1, but 
            # i is not in self.axes
            return array_api.log(result) + array_api.max(Z, axis=self.axes, keepdims=False)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Z = node.inputs[0]
        # input_shape = Z.shape

        # # TODO:
        # max_elements_along_axes = Z.cached_data.max(axis=self.axes, keepdims=True)
        # max_elements_along_axes = Tensor(max_elements_along_axes, requires_grad=False)
        # max_elements_along_axes = broadcast_to(max_elements_along_axes, input_shape)

        # result = exp(Z - max_elements_along_axes)
        # result = result / broadcast_to(summation(result, axes=self.axes, keep_axes=True), input_shape)

        # # TODO:
        # indices_of_max_elements = Z.cached_data.argmax(axis=self.axes, keepdims=True)
        # ones = array_api.zeros(input_shape)
        # ones[array_api.unravel_index(indices_of_max_elements, input_shape)] = 1
        # ones = Tensor(ones, requires_grad=False)

        # result = result + ones
        # result = out_grad * result

        # return result


def logsumexp(a: Tensor, axes: Optional[Union[tuple, int]] = None, keep_axes: bool = False):
    return LogSumExp(axes=axes, keep_axes=keep_axes)(a)


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    # numpy do not need device argument
    kwargs = {"device": device} if array_api is not numpy else {}
    device = device if device else cpu()

    if not rand or "dist" not in rand:
        arr = array_api.full(shape, fill_value, dtype=dtype, **kwargs)
    else:
        if rand["dist"] == "normal":
            arr = array_api.randn(
                shape, dtype, mean=rand["mean"], std=rand["std"], **kwargs
            )
        if rand["dist"] == "binomial":
            arr = array_api.randb(
                shape, dtype, ntrials=rand["trials"], p=rand["prob"], **kwargs
            )
        if rand["dist"] == "uniform":
            arr = array_api.randu(
                shape, dtype, low=rand["low"], high=rand["high"], **kwargs
            )

    return Tensor.make_const(arr, requires_grad=requires_grad)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
