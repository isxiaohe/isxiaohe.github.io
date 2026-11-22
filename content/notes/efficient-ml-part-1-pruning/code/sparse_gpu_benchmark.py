import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured

assert torch.cuda.is_available(), "2:4 sparse kernel example requires CUDA"
N = K = 256                 # multiples of 64 for FP16 in the documented API
B = 1024
x = torch.randn(B, K, device="cuda", dtype=torch.float16)
w = torch.randn(N, K, device="cuda", dtype=torch.float16)

# Each four-element group keeps the two largest magnitudes.
g = w.reshape(N, K // 4, 4)
keep_idx = g.abs().topk(2, dim=-1).indices
mask = torch.zeros_like(g).scatter_(-1, keep_idx, 1).reshape(N, K)
w_dense_zeroed = w * mask
w_packed = to_sparse_semi_structured(w_dense_zeroed)

with torch.inference_mode():
    yd = F.linear(x, w_dense_zeroed)  # dense GEMM, despite 50% zeros
    ys = F.linear(x, w_packed)        # supported sparse GEMM path
    print("max abs error:", (yd - ys).abs().max().item())

    def ms_per_call(fn, iterations=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iterations

    print("dense ms:", ms_per_call(lambda: F.linear(x, w_dense_zeroed)))
    print("2:4   ms:", ms_per_call(lambda: F.linear(x, w_packed)))
