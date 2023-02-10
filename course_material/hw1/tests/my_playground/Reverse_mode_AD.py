import sys
# append the "needle" path, so you can import needle
sys.path.append(r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw1\python")
import needle as ndl


v1 = ndl.Tensor([0], dtype="float32")
v2 = ndl.exp(v1)
v3 = v2 + 1
v4 = v2 * v3
print(v4)
print(v4.op)
print(v4.op.gradient.__code__)

v4_bar = ndl.Tensor([1], dtype="float32")
in_grads = v4.op.gradient(v4_bar, v4)

# v2_to_4_bar.inputs are (v4_bar, v3)
print(in_grads[0].inputs)