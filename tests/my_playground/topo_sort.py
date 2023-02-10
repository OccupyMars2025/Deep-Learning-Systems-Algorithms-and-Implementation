import sys
# append the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl

import numpy as np

# # Test case 1
# a1, b1 = ndl.Tensor(np.asarray([[0.88282157]])), ndl.Tensor(np.asarray([[0.90170084]]))
# c1 = 3*a1*a1 + 4*b1*a1 - a1

# soln = np.array([np.array([[0.88282157]]),    # a1
#                     np.array([[2.64846471]]), # 3*a1
#                     np.array([[2.33812177]]), # 3*a1*a1
#                     np.array([[0.90170084]]), # b1
#                     np.array([[3.60680336]]), # 4*b1
#                     np.array([[3.1841638]]),  # 4*b1*a1
#                     np.array([[5.52228558]]), # 3*a1*a1 + 4*b1*a1
#                     np.array([[-0.88282157]]),  # -a1
#                     np.array([[4.63946401]])]) # c1

# topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c1])])
# assert len(soln) == len(topo_order)
# assert soln.shape == topo_order.shape
# for i in range(len(soln)):
#     print(soln[i], topo_order[i], 20*'=', sep='\n\n')

# a1, b1 = ndl.Tensor([2]), ndl.Tensor([3])
# # c1 = a1 - b1
# c1 = a1 @ b1


# # Test case 2
# # a1.shape = (2, 1), b1.shape = (1, 2)
# a1, b1 = ndl.Tensor(np.asarray([[0.20914675], [0.65264178]])), ndl.Tensor(np.asarray([[0.65394286, 0.08218317]]))
# c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

# soln = [np.array([[0.65394286, 0.08218317]]),  # b1
#         np.array([[0.20914675], [0.65264178]]), # a1
#         np.array([[0.19040619]]),             # b1 @ a1
#         np.array([[1.53101102, 0.19240724]]), # 2.3412 * b1
#         np.array([[0.44577898]]),             # (2.3412 * b1) @ a1
#         np.array([[0.63618518]]),             # (b1 @ a1) + (2.3412 * b1) @ a1
#         np.array([[1.90855553]]),             # 3 * ((b1 @ a1) + (2.3412 * b1) @ a1)
#         np.array([[3.40855553]])]             # c1

# topo_order_by_inspection = [
#     b1,
#     a1,
#     b1 @ a1,
#     2.3412 * b1, 
#     (2.3412 * b1) @ a1,
#     (b1 @ a1) + (2.3412 * b1) @ a1,
#     3 * ((b1 @ a1) + (2.3412 * b1) @ a1),
#     c1
# ]

# topo_order = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

# assert len(soln) == len(topo_order)
# # step through list as entries differ in length
# for t, s, t2 in zip(topo_order, soln, topo_order_by_inspection):
#     np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)
#     np.testing.assert_allclose(t2.numpy(), s, rtol=1e-06, atol=1e-06)

# print("hello")


# # Test case 1
# a1, b1 = ndl.Tensor(np.asarray([[0.88282157]])), ndl.Tensor(np.asarray([[0.90170084]]))
# c1 = 3*a1*a1 + 4*b1*a1 - a1

# soln = np.array([np.array([[0.88282157]]), # a1
#                     np.array([[2.64846471]]), # 3*a1
#                     np.array([[2.33812177]]), # 3*a1*a1
#                     np.array([[0.90170084]]), # b1
#                     np.array([[3.60680336]]), # 4*b1
#                     np.array([[3.1841638]]),  # 4*b1*a1
#                     np.array([[5.52228558]]), # 3*a1*a1 + 4*b1*a1
#                     np.array([[-0.88282157]]),  # -a1
#                     np.array([[4.63946401]])]) # c1

# topo_order_by_inspection = [
#     a1,
#     3*a1,
#     3*a1*a1,
#     b1,
#     4*b1,
#     4*b1*a1,
#     3*a1*a1 + 4*b1*a1,
#     -a1,
#     c1
# ]
# for i in range(len(topo_order_by_inspection)):
#     topo_order_by_inspection[i] = topo_order_by_inspection[i].numpy()
# topo_order_by_inspection = np.array(topo_order_by_inspection)

# topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c1])])

# assert len(soln) == len(topo_order)
# assert soln.shape == topo_order_by_inspection.shape

# np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)
# np.testing.assert_allclose(topo_order_by_inspection, soln, rtol=1e-06, atol=1e-06)



# Test case 3
a = ndl.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
b = ndl.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
e = (a@b + b - a)@a

topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([e])])

soln = np.array([np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),   # a  
                np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]), # b
                np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),   # a@b  
                np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),  # a@b + b
                np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),  # -a
                np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),   # a@b + b - a
                np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]])])  # e

topo_order_by_inspection = [
    a, 
    b, 
    a@b,
    a@b + b,
    -a,
    a@b + b - a,
    e
]

for i in range(len(topo_order_by_inspection)):
    topo_order_by_inspection[i] = topo_order_by_inspection[i].numpy()
topo_order_by_inspection = np.array(topo_order_by_inspection)

assert len(soln) == len(topo_order)
assert soln.shape == topo_order_by_inspection.shape

np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)
np.testing.assert_allclose(topo_order_by_inspection, soln, rtol=1e-06, atol=1e-06)
print("case 3")
