import sys
# append the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl

import numpy as np


x = ndl.Tensor(np.random.randn())
print(x, type(x), x.shape, type(x.shape), len(x.shape), sep='\n')
print(40*"=")

x = ndl.Tensor([99])
print(x, type(x), x.shape, type(x.shape), len(x.shape), sep='\n')
