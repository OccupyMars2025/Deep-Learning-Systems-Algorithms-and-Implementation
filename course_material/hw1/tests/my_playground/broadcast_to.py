# # import numpy as np


# # x = np.array([1, 2, 3])
# # x2 = np.broadcast_to(x, (3, 3))
# # print(x)
# # print(x2)
# # print(id(x), id(x2), id(x) == id(x2), sep='\n')
# # print(x is x2)
# # print(20*"=")

# # try:
# #     x2[1, 0] = 99
# #     print(x, x2, sep='\n')
# # except ValueError as e:
# #     print("Error!", e)
# # print(30*"=")

# # x[0] = 99
# # print(x, x2, sep='\n')

# import torch

# x = torch.tensor([1, 2, 3])
# x2 = torch.broadcast_to(x, (3, 3))
# print(x, x2, sep='\n')
# print(id(x), id(x2), id(x) == id(x2), sep='\n')
# print(x is x2)
# print(20*"=")

# x[0] = 99
# print(x, x2, sep='\n')
# print(30*"=")

# try:
#     x2[1, 1] = 777
#     print(x, x2, sep='\n')
# except ValueError as e:
#     print("Error!", e)
# print(30*"=")

# x = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
# y = torch.broadcast_to(x, (3, 3))
# out_gradient = torch.randn(3, 3)
# out_gradient.requires_grad = True
# print(out_gradient)
# y.backward(out_gradient)
# print(x)
# print(x.grad)
# print(torch.sum(out_gradient, 0))
#
# import torch
#
# original_shape = (2, 3, 1, 9)
# broadcast_shape = (7, 8, 1, 2, 3, 4, 9)
# x = torch.randn(original_shape, requires_grad=True)
# # print(x)
# x2 = torch.broadcast_to(x, broadcast_shape)
# # print(x2)
# gradient = torch.randn(broadcast_shape)
# # print("gradient w.r.t x2", gradient, sep='\n')
# x2.backward(gradient)
# # print("gradient w.r.t x", x.grad, sep='\n')
#
# axes_to_sum_over = []
# for i in range(-1, -len(x2.shape) - 1, -1):
#     if i >= -len(x.shape):
#         if 1 == x.shape[i]:
#             axes_to_sum_over.append(i)
#     else:
#         axes_to_sum_over.append(i)
# assert torch.all(x.grad == torch.sum(gradient, dim=axes_to_sum_over, keepdim=True))
# print("Successful!!!")

import torch

x = torch.tensor([1, 2, 3])
print(id(x))
x1 = torch.broadcast_to(x, [3, 3])
sum = torch.sum(x1)
x[0] += 1
print(id(x))
print(x)
print(x1)
sum2 = torch.sum(x1)
print(sum2, sum)
assert (sum2 - sum) == 3