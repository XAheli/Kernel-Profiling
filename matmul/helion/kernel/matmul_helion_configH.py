#!/usr/bin/env python3
"""NCU profiling: Helion batched matmul, Config H (B=24, M=48K, K=2048, N=2048, FP32)"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import helion
import helion.language as hl

# ── Config H ──
BATCH, M, K, N = 24, 48_000, 2_048, 2_048
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
    print(f"Config H | Helion | matmul ({BATCH},{M},{K})x({BATCH},{K},{N}) | {DTYPE}")
    elem_bytes = 4
    required = (BATCH * M * K + BATCH * K * N + BATCH * M * N) * elem_bytes
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"GPU memory: {free_mem/1e9:.1f} GB free / {total_mem/1e9:.1f} GB total")
    print(f"Required:   {required/1e9:.1f} GB")
    if required > free_mem * 0.95:
        print("ERROR: Insufficient GPU memory")
        exit(1)

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
