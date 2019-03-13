'''
Converting a Tensor to a NumPy array and vice versa is easy.
The Torch Tensor and NumPy array will share their underlying memory
locations and changing one will change the other.
'''
import torch
a = torch.ones(5)
b = a.numpy()
print(a)
print(b)
a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
'''OUTPUT:
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)'''
