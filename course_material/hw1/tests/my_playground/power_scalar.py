import sys
# append a the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")

import needle as ndl

x = ndl.Tensor([1, 2, 3], dtype='float32')
y = ndl.ops.power_scalar(x, 2.2)
print(y, y.dtype, y.shape)
