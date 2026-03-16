#!/usr/bin/env python3
"""NCU profiling: Triton vector-add, Mixed (B=64, S=16K, H=2048, FP16)"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import triton
import triton.language as tl

# -- Mixed MP --
BATCH, SEQ, HIDDEN = 64, 16_000, 2_048
DTYPE = torch.float16
N = BATCH * SEQ * HIDDEN

# -- Triton kernel --
@triton.jit
def _triton_vadd_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)

def triton_vadd(a, b):
    c = torch.empty_like(a)
    n = c.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _triton_vadd_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)
    return c

if __name__ == "__main__":
    print(f"MixedMP | Triton | N={N:,} | {DTYPE}")
    if N > 2**31 - 1:
        print(f"Warning: N={N:,} exceeds int32 max, using int64 indexing")
    a = torch.ones(N, device="cuda", dtype=DTYPE)
    b = torch.full((N,), 2.0, device="cuda", dtype=DTYPE)

    for _ in range(2):
        c = triton_vadd(a, b)
    torch.cuda.synchronize()

    c = triton_vadd(a, b)
    torch.cuda.synchronize()

    assert torch.allclose(c, torch.full_like(c, 3.0), atol=1e-3), "FAIL"
    print("Correctness: PASS")
