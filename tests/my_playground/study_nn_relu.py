import sys
# append the "needle" path, so you can import needle
sys.path.append("python")
import needle as ndl
import numpy as np

f = ndl.nn.ReLU()
x = ndl.Tensor(np.random.randint(-100, 100, (2, 4)))
print(x)
y = f(x)
print(y)