import torch

# Calibration activations X: [B, K]; weights W: [N, K]
B, K, N = 128, 16, 8
X = torch.randn(B, K)
W = torch.randn(N, K)

magnitude = W.abs()
input_scale = X.square().sum(dim=0).sqrt()  # [K]
wanda = magnitude * input_scale[None, :]    # [N, K]

# Keep the top 50% in each output row
keep = wanda.topk(K // 2, dim=-1).indices
mask = torch.zeros_like(W)
mask.scatter_(1, keep, 1.0)

assert (mask == 0).float().mean().item() == 0.5
