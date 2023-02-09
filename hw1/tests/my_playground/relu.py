import sys
# append a the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl
import numpy as np

# x = np.random.randn(3, 4)
# x[x < 0] = 0
# print(x)

# x = np.random.randn(3, 4)
# x2 = np.array(x)
# print(x2 is x)

# x_np = np.random.randn(3, 4)
# x_needle = ndl.Tensor(x_np)
# y_needle = ndl.relu(x_needle)
# # print(y_needle)
# # print(x_np)
# # print(x_needle.numpy() is x_np)
# # print(y_needle.numpy() is x_needle.numpy())
# # print(y_needle is x_needle)
# y_needle.backward(ndl.Tensor(np.random.randn(3, 4)))
# print(x_needle.numpy())
# print(x_needle.grad)


# a = np.random.randint(-100, 100, (2, 3))
# print(a)
# b = np.where(a < 0, 0, a)
# print(b is a)
# print(b)

# x = np.arange(4)
# # x = np.arange(4).astype("float32")
# print(x, x.dtype)
# for i in range(x.size):
#     # x.flat[i] += 0.5
#     x.flat[i] += 1e-4
# print(x, x.dtype)

x = ndl.Tensor(np.random.randint(-100, 100, (2, 3)))
x1 = x > 0
print(x1)