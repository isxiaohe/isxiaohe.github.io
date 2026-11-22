import torch
X = torch.tensor([[1., 1.], [1., 1.1]], dtype=torch.float64)
w = torch.tensor([1., 1.], dtype=torch.float64)
y = X @ w
naive = torch.tensor([1., 0.], dtype=torch.float64)
w1 = torch.linalg.lstsq(X[:, :1], y).solution.item()
refit = torch.tensor([w1, 0.], dtype=torch.float64)
err = lambda z: 0.5 * ((X @ z - y) ** 2).sum().item()
print(f"naive = {err(naive):.4f}; refit = {err(refit):.4f}; w1 = {w1:.2f}")
# naive = 1.1050; refit = 0.0025; w1 = 2.05
