"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


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
    def __init__(self, axes: Optional[tuple] = None, keep_axes: bool = False):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes, )
        else:
            self.axes = axes
        self.keep_axes = keep_axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes, keepdims=self.keep_axes)

    def gradient(self, out_grad, node):
        original_shape = node.inputs[0].shape
        if self.axes and not self.keep_axes:
            shape_that_keeps_axes = list(original_shape)
            for i in self.axes:
                shape_that_keeps_axes[i] = 1
            temp = reshape(out_grad, shape_that_keeps_axes)
            return broadcast_to(temp, original_shape)
        else:
            return broadcast_to(out_grad, original_shape)


def summation(a: Tensor, axes=None, keep_axes=False):
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

