import torch
from torch import nn

conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

# Keep five filters with the largest L2 norm.
score = conv1.weight.detach().flatten(1).norm(dim=1)
keep = score.topk(5).indices.sort().values
small1 = nn.Conv2d(3, 5, 3, padding=1)
small2 = nn.Conv2d(5, 16, 3, padding=1)
with torch.no_grad():
    small1.weight.copy_(conv1.weight[keep])
    small1.bias.copy_(conv1.bias[keep])
    small2.weight.copy_(conv2.weight[:, keep])  # crucial downstream slice
    small2.bias.copy_(conv2.bias)

x = torch.randn(2, 3, 32, 32)
y = small2(torch.relu(small1(x)))
assert y.shape == (2, 16, 32, 32)
