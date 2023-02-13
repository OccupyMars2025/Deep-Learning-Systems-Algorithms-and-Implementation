import sys
sys.path.append("./python")
import needle as ndl

import numpy as np

# x = np.random.rand(2, 3)
# print(x)
# print(x.dtype)
# x2 = x.astype("float32")
# print(x.dtype)
# print(x2.dtype)
# print(x is x2)
# print(id(x), id(x2), sep='\n')

# x = np.ones((2, 3), dtype="int32")
# print(x, x.dtype)
# x2 = x * 10.1
# print(x2, x2.dtype)


# w = ndl.init.randb(2, 3, p=0.8)
# print(w, w.dtype)

# w = ndl.init.one_hot(n = 6, i = ndl.Tensor([5, 2, 0, 3]))
# print(w, w.dtype)

bias = ndl.init.kaiming_uniform(fan_in=4, fan_out=1, requires_grad=True)
print(bias)