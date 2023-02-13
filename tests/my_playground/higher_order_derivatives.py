import sys
# append a the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl
import numpy as np

# 0 <= x3 <= 1
x1, x2, x3 = ndl.Tensor(np.random.randn()), ndl.Tensor(np.random.randn()), ndl.Tensor(np.random.rand())
y = ndl.exp(x1 * x2) - ndl.log(x2**2 + 10) + x1/(x3 + 10)

#==================first-order derivatives : start =======================
y.backward()

grad_x1 = x2 * ndl.exp(x1 * x2) + ndl.Tensor(1) / (x3 + 10)
np.testing.assert_allclose(x1.grad.numpy(), grad_x1.numpy())

grad_x2 = x1 * ndl.exp(x1 * x2) - (2 * x2) / (x2**2 + 10)
np.testing.assert_allclose(x2.grad.numpy(), grad_x2.numpy())

grad_x3 = -x1 / (x3 + 10)**2
np.testing.assert_allclose(x3.grad.numpy(), grad_x3.numpy())

print("grad_x1, grad_x2, grad_x3: \n", x1.grad, x2.grad, x3.grad)
print(grad_x1, grad_x2, grad_x3)

#==================first-order derivatives : end =======================

#==================second-order derivatives : start =======================
# Now, x2.grad is grad_x2 (that is the gradient of y w.r.t x2)
x2.grad.backward()

grad_x2_x1 = ndl.exp(x1 * x2) * (1 + x1 * x2)
np.testing.assert_allclose(x1.grad.numpy(), grad_x2_x1.numpy())

grad_x2_x2 = (x1 ** 2) * ndl.exp(x1 * x2) - (20 - 2*x2**2) / (x2**2 + 10)**2
# np.testing.assert_allclose(x2.grad.numpy(), grad_x2_x2.numpy(), rtol=1e-3)

grad_x2_x3 = 0

print("grad_x2_x1, grad_x2_x2, grad_x2_x3: \n", x1.grad, x2.grad, x3.grad)
print(grad_x2_x1, grad_x2_x2, grad_x2_x3)




