import torch
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print('x:', x, '\n', 'y:', y)
print('x + y', x + y)   #syntax torch.add(x, y)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

print('y.add_(x)\n',y.add_(x))  # Any operation that mutates a tensor \
# in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), \
# will change x.

print(x[:, 1])  # can use standard NumPy-like indexing

# view = reshape, but doesn't change the original
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())  # notice that dont change x

# If you have a one element tensor, use .item() \
# to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

'''
OUTPUT

x: tensor([[0.7314, 0.5876, 0.3133],
        [0.6724, 0.1104, 0.5805],
        [0.3305, 0.4698, 0.6518],
        [0.8331, 0.1495, 0.7435],
        [0.2830, 0.2685, 0.9160]]) 
 y: tensor([[0.4574, 0.5324, 0.6722],
        [0.5772, 0.2691, 0.7995],
        [0.5190, 0.9900, 0.2968],
        [0.8865, 0.7363, 0.3505],
        [0.0270, 0.3050, 0.8132]])
x + y tensor([[1.1887, 1.1200, 0.9855],
        [1.2497, 0.3795, 1.3799],
        [0.8496, 1.4599, 0.9487],
        [1.7195, 0.8858, 1.0939],
        [0.3099, 0.5735, 1.7292]])
        
tensor([[1.1887, 1.1200, 0.9855],
        [1.2497, 0.3795, 1.3799],
        [0.8496, 1.4599, 0.9487],
        [1.7195, 0.8858, 1.0939],
        [0.3099, 0.5735, 1.7292]])
        
y.add_(x)
 tensor([[1.1887, 1.1200, 0.9855],
        [1.2497, 0.3795, 1.3799],
        [0.8496, 1.4599, 0.9487],
        [1.7195, 0.8858, 1.0939],
        [0.3099, 0.5735, 1.7292]])
        
tensor([0.5876, 0.1104, 0.4698, 0.1495, 0.2685])
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
tensor([0.0943])
0.09433865547180176
'''
