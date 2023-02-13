import sys
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems-Algorithms-and-Implementation\python")

import math
from needle import init
import numpy as np


# fan_in = 1
# gain = 2**0.5
# bound = gain * (3 / fan_in)**0.5
# print(bound)

# bound_2 = math.pow(2, 0.5) * math.pow((3/ fan_in), 0.5)
# print(bound_2)

out_features = 4
device=None
dtype="float32"

np.random.seed(1337)
"""
E       
E       Mismatched elements: 4 / 4 (100%)
E       Max absolute difference: 1.1202965
E       Max relative difference: 1.0000094
E        x: array([[ 0.155295,  1.628277, -1.541949,  2.240593]], dtype=float32)
E        y: array([[ 0.077647,  0.814139, -0.770975,  1.120297]], dtype=float32)
"""
bias = init.kaiming_uniform(1, out_features, device=device, dtype=dtype, requires_grad=True)
print(bias.cached_data)