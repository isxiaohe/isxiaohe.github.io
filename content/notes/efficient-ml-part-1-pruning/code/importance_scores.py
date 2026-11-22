import torch
# Calibration activations X: [B, K]; linear weight W: [N, K].
B, K, N = 128, 16, 8
X, W = torch.randn(B, K), torch.randn(N, K)
magnitude = W.abs()
wanda = W.abs() * X.square().sum(dim=0).sqrt()[None, :]

# Per-output, 50%-sparse mask from Wanda scores.
keep = wanda.topk(k=K // 2, dim=1).indices
mask = torch.zeros_like(W).scatter_(1, keep, 1)
assert (mask == 0).float().mean() == 0.5
