"""
The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        # return []
        raise NotImplementedError()


# 2023/2/11 18:00 : It seems that '-> List["Module"]' cannot be changed to '-> List[Module]' 
# in the following line. Why ???  
def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    elif isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        # return []
        raise NotImplementedError()


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    # def forward(self, *args, **kwargs):
    #     raise NotImplementedError()


class Identity(Module):
    def forward(self, x):
        return x

# Caution :
# 1. Be careful to explicitly broadcast the bias term to 
# the correct shape -- Needle does not support implicit broadcasting.
# 2. Additionally note that, for all layers including this one, you should initialize 
# the weight Tensor before the bias Tensor, 
# and should initialize all Parameters using only functions from 'init'.
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        """
        in_features - size of each input sample
        out_features - size of each output sample
        bias - If set to False, the layer will not learn an additive bias.


        weight - the learnable weights of shape (in_features, out_features). The values should
              be initialized with the Kaiming Uniform initialization with fan_in = in_features
        bias - the learnable bias of shape (out_features). The values should be initialized 
              with the Kaiming Uniform initialized with fan_out = out_features. Note the different in 
             fan_in choice, due to their relative sizes.


        Applies a linear transformation to the incoming data:  Y = XA+b . The input shape is
               (N,H_in)  where  H_in=in_features . The output shape is  (N,H_out)  where  H_out=out_features .
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # You should initialize the weight Tensor before the bias Tensor, and should 
        # initialize all Parameters using only functions from 'init'. 
        # It can be explained as follows:
        # np.random.seed(2)
        # weight = np.random.rand(2, 3)
        # bias= np.random.rand(2, 3)

        # np.random.seed(2)
        # bias= np.random.rand(2, 3)
        # weight = np.random.rand(2, 3)
        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True)
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True)
            self.bias = self.bias.reshape([1, out_features])
        else:
            self.bias = None
        

    def forward(self, X: Tensor) -> Tensor:
        """
        X: shape = (N, in_features)

        Be careful to explicitly broadcast the bias term to the correct shape 
        -- Needle does not support implicit broadcasting
        """
        if self.bias:
            # return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
            return X.matmul(self.weight) + self.bias.broadcast_to((X.shape[0], self.out_features))
        else:
            # return ops.matmul(X, self.weight)
            return X.matmul(self.weight)


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
