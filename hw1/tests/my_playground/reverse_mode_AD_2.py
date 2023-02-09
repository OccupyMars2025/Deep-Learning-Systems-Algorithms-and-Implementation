import sys
# append a the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl
import numpy as np


# class MyClass:
#     grad: np.ndarray

# myclass = MyClass()
# print(myclass)
# print(myclass.__dict__)

# x = ndl.Tensor([1, 2, 3])
# x = x.sum()
# print(x)

# x = [ndl.Tensor([2, 3]), ndl.Tensor([20, 30]), ndl.Tensor([200, 300])]
# x = sum(x)
# print(x)

# x1 = np.random.randn(4, 5)
# y1 = np.linalg.norm(x1)
# y2 = ((x1**2).sum())**0.5
# np.testing.assert_allclose(y1, y2)
# print(y1)
# print(y2)

# for i in range(2, 10):
#     x = ndl.Tensor([i])
#     y = x * x
#     y.backward()
#     assert x.grad.numpy()[0] == 2*i

# print(y.grad, x.grad, sep='\n')

# # check gradient of gradient
# x2 = ndl.Tensor([3])
# x3 = ndl.Tensor([4])
# y = x2 * x2 + x2 * x3
# topological_sort = ndl.autograd.find_topo_sort([y])
# print(topological_sort)
# print("len(topological_sort): ", len(topological_sort))
# y.backward()
# grad_x2 = x2.grad
# grad_x3 = x3.grad
# print(x2.grad, x3.grad)
# # gradient of gradient
# topological_sort_2 = ndl.autograd.find_topo_sort([grad_x2])
# print(topological_sort_2)
# print("len(topological_sort_2): ", len(topological_sort_2))

# grad_x2.backward()

# topological_sort_3 = ndl.autograd.find_topo_sort([x2])
# print(topological_sort_3)
# print("len(topological_sort_3): ", len(topological_sort_3))

# grad_x2_x2 = x2.grad
# grad_x2_x3 = x3.grad
# x2_val = x2.numpy()
# x3_val = x3.numpy()
# assert y.numpy() == x2_val * x2_val + x2_val * x3_val
# assert grad_x2.numpy() == 2 * x2_val + x3_val
# assert grad_x3.numpy() == x2_val
# assert grad_x2_x2.numpy() == 2
# assert grad_x2_x3.numpy() == 1