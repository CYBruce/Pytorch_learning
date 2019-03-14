'''The autograd package provides automatic differentiation
for all operations on Tensors. It is a define-by-run
framework, which means that your backprop is defined by
how your code is run, and that every single iteration
can be different.'''
import torch
import numpy as np
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)  # tensor([[3., 3.], [3., 3.]], grad_fn=<AddBackward0>)
print(y.grad_fn)  # <AddBackward0 object at 0x10908fd30>

z = y * y * 3  # tensor([[27., 27.],[27., 27.]], grad_fn=<MulBackward0>)
out = z.mean()   # tensor(27., grad_fn=<MeanBackward1>)

a = torch.rand(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False
a.requires_grad_(True)
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)  # <SumBackward0 object at 0x10c56e438>

# Compute gradient
# because out contains a single scalar, out.backward()
# is equivalent to out.backward(torch.tensor(1.))
out.backward()
print(x.grad)
print(y.requires_grad)
print(y.grad)  # y.grad = None, can use register_hook function
print(y.grad_fn)

#Vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2  # y.grad_fn=<MulBackward0>
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
'''Now in this case y is no longer a scalar. torch.autograd
could not compute the full Jacobian directly, but if we
just want the vector-Jacobian product, simply pass the
vector to backward as argument:'''
y.backward(v)
print(x.grad)

'''You can also stop autograd from tracking history on
Tensors with .requires_grad=True by wrapping the
code block in with torch.no_grad():'''
print(x.requires_grad)  # True
print((x ** 2).requires_grad)  # True

with torch.no_grad():
    print((x ** 2).requires_grad)  # False
