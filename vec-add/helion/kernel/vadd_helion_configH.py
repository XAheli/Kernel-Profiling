#!/usr/bin/env python3
"""NCU profiling: Helion vector-add, Config H (B=24, S=48K, H=2048, FP32)"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import helion
import helion.language as hl

# ── Config H ──
BATCH, SEQ, HIDDEN = 24, 48_000, 2_048
DTYPE = torch.float32
N = BATCH * SEQ * HIDDEN

# ── Helion kernel ──
@helion.kernel(config=helion.Config(block_sizes=[1024]))
def _helion_vadd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile_n in hl.tile(a.size()):
        a_tile = a[tile_n]
        b_tile = b[tile_n]
        out[tile_n] = a_tile + b_tile
    return out

def helion_vadd(a, b):
    return _helion_vadd(a, b)

if __name__ == "__main__":
    print(f"Config H | Helion | N={N:,} | {DTYPE}")
    free_mem, total_mem = torch.cuda.mem_get_info()
    required = 3 * N * 4
    print(f"GPU memory: {free_mem/1e9:.1f} GB free / {total_mem/1e9:.1f} GB total")
    print(f"Required:   {required/1e9:.1f} GB")
    if required > free_mem * 0.95:
        print("ERROR: Insufficient GPU memory")
        exit(1)

    a = torch.ones(N, device="cuda", dtype=DTYPE)
    b = torch.full((N,), 2.0, device="cuda", dtype=DTYPE)

    for _ in range(2):
        c = helion_vadd(a, b)
    torch.cuda.synchronize()

    c = helion_vadd(a, b)
    torch.cuda.synchronize()

    assert torch.allclose(c, torch.full_like(c, 3.0), atol=1e-5), "FAIL"
    print("Correctness: PASS")
