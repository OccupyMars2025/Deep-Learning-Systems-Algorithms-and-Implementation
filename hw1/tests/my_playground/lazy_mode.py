import sys
# append a the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl

ndl.autograd.LAZY_MODE = True
x1 = ndl.Tensor([3], dtype="float32")
x2 = ndl.Tensor([4], dtype="float32")
x3 = x1 * x2
print(x3.cached_data is None)
print(x3)
print(x3.cached_data is None)
