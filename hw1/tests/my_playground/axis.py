import numpy as np

x = np.random.randn(2, 3, 4)
x1 = np.sum(x, axis=[-1, -2], keepdims=True)
print(np.__version__)
print(x1)