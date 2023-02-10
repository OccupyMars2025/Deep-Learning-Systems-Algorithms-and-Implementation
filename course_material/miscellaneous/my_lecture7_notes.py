import sys
# append a the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl
import numpy as np

v1 = ndl.Tensor([1])
v2 = ndl.exp(v1)
v3 = v2 + 1
v4 = v2 * v3

# TODO: How should "backward()" deal with "if-else" statement ???
if v4.numpy() > 0.5:
    v5 = v4 * 2
else:
    v5 = v4
v5.backward()

print(v5)
print(v1.grad)