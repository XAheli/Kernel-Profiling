#!/usr/bin/env python3
"""NCU profiling: Helion batched matmul, Config G (B=48, M=12K, K=1536, N=1536, FP32)"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import helion
import helion.language as hl

# ── Config G ──
BATCH, M, K, N = 48, 12_000, 1_536, 1_536
DTYPE = torch.float32

# ── Helion kernel ──
@helion.kernel(static_shapes=True)
def run_helion_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty([b, m, n], device=A.device, dtype=A.dtype)

    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n])
        out[tile_b, tile_m, tile_n] = acc.to(A.dtype)
    return out

if __name__ == "__main__":
    print(f"Config G | Helion | matmul ({BATCH},{M},{K})x({BATCH},{K},{N}) | {DTYPE}")
    A = torch.randn(BATCH, M, K, device="cuda", dtype=DTYPE)
    B = torch.randn(BATCH, K, N, device="cuda", dtype=DTYPE)

    for _ in range(2):
        C = run_helion_matmul(A, B)
    torch.cuda.synchronize()

    C = run_helion_matmul(A, B)
    torch.cuda.synchronize()

    ref = torch.bmm(A, B)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-3), \
        f"FAIL: max diff = {(C - ref).abs().max().item()}"
    print("Correctness: PASS")
