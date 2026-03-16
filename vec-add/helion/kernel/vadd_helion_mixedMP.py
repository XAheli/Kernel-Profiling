#!/usr/bin/env python3
"""NCU profiling: Helion vector-add, Mixed (B=64, S=16K, H=2048, FP16)"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import helion
import helion.language as hl

# ── Mixed MP ──
BATCH, SEQ, HIDDEN = 64, 16_000, 2_048
DTYPE = torch.float16
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
    print(f"MixedMP | Helion | N={N:,} | {DTYPE}")
    a = torch.ones(N, device="cuda", dtype=DTYPE)
    b = torch.full((N,), 2.0, device="cuda", dtype=DTYPE)

    for _ in range(2):
        c = helion_vadd(a, b)
    torch.cuda.synchronize()

    c = helion_vadd(a, b)
    torch.cuda.synchronize()

    assert torch.allclose(c, torch.full_like(c, 3.0), atol=1e-3), "FAIL"
    print("Correctness: PASS")
