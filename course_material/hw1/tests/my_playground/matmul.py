import sys
# append the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl

import numpy as np


x1_np = np.random.randn(5, 4)
x2_np = np.random.randn(6, 6, 4, 3)
y_ndl = ndl.matmul(ndl.Tensor(x1_np), ndl.Tensor(x2_np))
# print(y_ndl)
print(y_ndl.shape)
gradient_wrt_y_ndl = ndl.Tensor(np.ones(y_ndl.shape))
gradient_wrt_x1_ndl, gradient_wrt_x2_ndl = y_ndl.op.gradient(gradient_wrt_y_ndl, y_ndl)
print(gradient_wrt_x1_ndl.shape, gradient_wrt_x2_ndl.shape)

print((5, 2) == (5, 3))
print(tuple(range(-1, -5, -1)))