import math
import torch
from torch import nn

class TinyMHA(nn.Module):
    def __init__(self, d_model=32, n_heads=4, d_head=8):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):  # [B, T, Dmodel]
        b, t, _ = x.shape
        proj = lambda layer: layer(x).reshape(b, t, self.n_heads, self.d_head)
        q, k, v = proj(self.q), proj(self.k), proj(self.v)
        attn = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.d_head)
        z = torch.einsum("bhts,bshd->bthd", attn.softmax(-1), v)
        return self.out(z.reshape(b, t, -1))

@torch.no_grad()
def keep_heads(old, heads):
    d_model = old.q.in_features
    new = TinyMHA(d_model, len(heads), old.d_head).to(old.q.weight.device)
    idx = torch.tensor([h * old.d_head + d for h in heads
                        for d in range(old.d_head)], device=old.q.weight.device)
    for name in ("q", "k", "v"):
        getattr(new, name).weight.copy_(getattr(old, name).weight[idx, :])
    new.out.weight.copy_(old.out.weight[:, idx])
    return new

old = TinyMHA()
small = keep_heads(old, [0, 2, 3])
assert small(torch.randn(2, 5, 32)).shape == (2, 5, 32)
print(old.q.weight.shape, small.q.weight.shape)  # [32,32] -> [24,32]
