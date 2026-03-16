#!/usr/bin/env python3
"""NCU profiling: Helion vector-add, Config G (B=48, S=12K, H=1536, FP32)"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import helion
import helion.language as hl

# ── Config G ──
BATCH, SEQ, HIDDEN = 48, 12_000, 1_536
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
    print(f"Config G | Helion | N={N:,} | {DTYPE}")
    a = torch.ones(N, device="cuda", dtype=DTYPE)
    b = torch.full((N,), 2.0, device="cuda", dtype=DTYPE)

    for _ in range(2):
        c = helion_vadd(a, b)
    torch.cuda.synchronize()

    c = helion_vadd(a, b)
    torch.cuda.synchronize()

    assert torch.allclose(c, torch.full_like(c, 3.0), atol=1e-5), "FAIL"
    print("Correctness: PASS")
