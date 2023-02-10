# import numdifftools as nd

# print(nd.__version__)

# nd.test('--doctest-modules', '--disable-warnings')

# import numpy as np
# import numdifftools as nd
# f = nd.Derivative(np.exp, full_output=True)
# print(f)
# val, info = f(0)
# print(val, info)
# np.testing.assert_allclose(val, 1)
# # np.testing.assert_allclose(info.error_estimate, 5.28466160e-14)


# import numpy as np
# import numdifftools as nd
# df = nd.Derivative(np.sin, n=1)
# print(df(0))
# print(np.allclose(df(0), 1.))

# # -sin(x)
# ddf = nd.Derivative(np.sin, n=2)
# print(ddf(0))
# print(np.allclose(ddf(0), 0.))

# # -cos(x)
# dddf = nd.Derivative(np.sin, n=3)
# np.testing.assert_allclose(dddf(0), -1.)

# # sin(x)
# ddddf = nd.Derivative(np.sin, n=4)
# np.testing.assert_allclose(ddddf(0), 0.)


# import numpy as np
# import numdifftools as nd
# import matplotlib.pyplot as plt
# x = np.linspace(-2, 2, 100)
# for i in range(10):
#     df = nd.Derivative(np.tanh, n=i)
#     y = df(x)
#     h = plt.plot(x, y/np.abs(y).max())
# plt.show()


import numpy as np
import numdifftools as nd

def rosen(x): 
    return (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2

grad = nd.Gradient(rosen)([1, 1])

print(grad)
np.testing.assert_allclose(grad, 0)

H = nd.Hessian(rosen)([1, 1])
print(H, type(H))

li, U = np.linalg.eig(H)
print(li, U, sep='\n')