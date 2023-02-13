"""
Following the same methodology of the existing initializers (you will want to call e.g. 
init.rand or init.randn from your functions below, implement the following common 
initialization methods. In all cases, the functions should return 
fan_in by fan_out 2D tensors (extensions to other sizes can be done via e.g., reshaping).
"""
import math
import needle as ndl

from typing import Optional

def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = ndl.cpu() if device is None else device
    #TODO: Why don't you specify "dtype" here ?  2023/2/11 14:00
    array = device.rand(*shape) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
    

def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = ndl.cpu() if device is None else device
    #TODO: Why don't you specify "dtype" here ?  2023/2/11 14:00
    array = device.randn(*shape) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = ndl.cpu() if device is None else device
    # array.dtype may be different with "dtype"
    array = device.ones(*shape, dtype=dtype) * c # note: can change dtype
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(*shape, c=1, device=device, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(*shape, c=0, device=device, dtype=dtype, requires_grad=requires_grad)


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n: int, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor 
        "n" is similar to the number of classes.
        len(i.shape) should be 1 .

        i: ndl.Tensor
    """
    device = ndl.cpu() if device is None else device
    return ndl.Tensor(device.one_hot(n,i.numpy(), dtype=dtype), device=device, requires_grad=requires_grad)


def xavier_uniform(fan_in: int, fan_out: int, gain=1.0, device = None, dtype="float32", requires_grad=False):
    """
    fan_in - dimensionality of input
    fan_out - dimensionality of output
    gain - optional scaling factor
    return: a new needle.Tensor with shape (fan_in, fan_out)

     device: Optional[ndl.autograd.Device]

    Fills the input Tensor with values according to the method described in "Understanding the difficulty of 
    training deep feedforward neural networks", using a uniform distribution. 

    Pass remaining **kwargs parameters to the corresponding init random call.
    """
    a = gain * (6 / (fan_in + fan_out))**0.5
    return rand(fan_in, fan_out, low=-a, high=a, device=device, dtype=dtype, requires_grad=requires_grad)


def xavier_normal(fan_in: int, fan_out: int, gain=1.0, device = None, dtype="float32", requires_grad=False):
    """
    Fills the input Tensor with values according to the method described in 
    [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    , using a normal distribution. 

    ##### Parameters
    - `fan_in` - dimensionality of input
    - `fan_out` - dimensionality of output
    - `gain` - optional scaling factor
    return: a new needle.Tensor with shape (fan_in, fan_out)

    device: Optional[ndl.autograd.Device] 

    """
    std = gain * (2 / (fan_in + fan_out))**0.5
    return randn(fan_in, fan_out, mean=0, std=std, device=device, dtype=dtype, requires_grad=requires_grad)



def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity="relu", device = None, dtype="float32", requires_grad=False):
    """
    fan_in - dimensionality of input
    fan_out - dimensionality of output
    nonlinearity - the non-linear function
    return: a new needle.Tensor with shape (fan_in, fan_out)

    device: Optional[ndl.autograd.Device] 

    Fills the input Tensor with values according to the method described in "Delving deep into rectifiers:
     Surpassing human-level performance on ImageNet classification", using a uniform distribution
    """
    assert nonlinearity == "relu", "Only relu supported currently"
    if nonlinearity == "relu":
        gain = 2**0.5
    else:
        raise NotImplementedError()
    bound = gain * (3 / fan_in)**0.5
    return rand(fan_in, fan_out, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=requires_grad)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", device=None, dtype="float32", requires_grad=False):
    """
    fan_in - dimensionality of input
    fan_out - dimensionality of output
    nonlinearity - the non-linear function
    return: a new needle.Tensor with shape (fan_in, fan_out)


    Fills the input Tensor with values according to the method described in "Delving deep into rectifiers:
     Surpassing human-level performance on ImageNet classification", 
    """
    assert nonlinearity == "relu", "Only relu supported currently"
    if nonlinearity == "relu":
        gain = 2**0.5
    else:
        raise NotImplementedError()
    std = gain/ (fan_in ** 0.5)

    return randn(fan_in, fan_out, mean=0, std=std, device=device, dtype=dtype, requires_grad=requires_grad)
