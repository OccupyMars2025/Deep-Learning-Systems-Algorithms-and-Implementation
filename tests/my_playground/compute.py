import sys
# append a the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")

import needle as ndl

x1 = ndl.Tensor([11], dtype="int32")
x2 = ndl.Tensor([22], dtype="int32")
x3 = x1 * x2
print(x3)
print(x3.op.compute)
print(x3.op.compute.__code__)
print(x3 + 1)
print(type(x3.cached_data))