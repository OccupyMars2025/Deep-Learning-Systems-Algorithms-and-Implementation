import sys
# append the "needle" path, so you can import needle
sys.path.append("./python")
import needle as ndl
import numpy as np

# w = ndl.Tensor([1, 2, 3], dtype="float32")
# grad = ndl.Tensor([1, 1, 1], dtype="float32")
# lr = 0.1
# for i in range(5):
#     w = w + (-lr) * grad

# print(w)
# w2 = w.data
# print(w2)
# print(id(w), id(w2), sep='\n')
# print(w.cached_data is w2.cached_data)
# w3 = w.data
# print(id(w3))
# print(w2.cached_data is w3.cached_data)

x = np.random.randn(2, 3)
print(x)
print(id(x))
x = x + 1
print(x)
print(id(x))