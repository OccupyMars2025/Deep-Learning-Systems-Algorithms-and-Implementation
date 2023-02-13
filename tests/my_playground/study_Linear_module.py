import sys
sys.path.append("python")
import needle as ndl
import numpy as np

"""
E       AssertionError: 
E       Not equal to tolerance rtol=1e-05, atol=1e-05
E       
E       Mismatched elements: 4 / 4 (100%)
E       Max absolute difference: 1.1202965
E       Max relative difference: 1.0000094
E        x: array([[ 0.155295,  1.628277, -1.541949,  2.240593]], dtype=float32)
E        y: array([[ 0.077647,  0.814139, -0.770975,  1.120297]], dtype=float32)

tests\test_nn_and_optim.py:542: AssertionError

"""


# def test_nn_linear_bias_init_1():
#     np.testing.assert_allclose(nn_linear_bias_init(),
#         np.array([[ 0.077647,  0.814139, -0.770975,  1.120297]],
#          dtype=np.float32), rtol=1e-5, atol=1e-5) 

# def nn_linear_bias_init():
#     np.random.seed(1337)
#     f = ndl.nn.Linear(7, 4)
#     return f.bias.cached_data

# test_nn_linear_bias_init_1()


# np.random.seed(2)
# weight = np.random.rand(2, 3)
# bias= np.random.rand(2, 3)

# print(weight, bias, sep='\n')
# array1 = np.array([[0.4359949,  0.02592623, 0.54966248],
#                    [0.43532239, 0.4203678,  0.33033482]])
# np.testing.assert_allclose(weight, array1)


# array2 = np.array([[0.20464863, 0.61927097, 0.29965467],
#                    [0.26682728, 0.62113383, 0.52914209]])
# np.testing.assert_allclose(bias, array2)