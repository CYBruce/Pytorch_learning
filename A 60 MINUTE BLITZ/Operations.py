import torch
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print('x:', x, '\n', 'y:', y)
print('x + y', x + y)   #syntax torch.add(x, y)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

print('y.add_(x)\n',y.add_(x))  #Any operation that mutates a tensor \
# in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), \
# will change x.
