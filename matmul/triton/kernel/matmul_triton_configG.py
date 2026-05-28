#!/usr/bin/env python3
"""NCU profiling: Triton batched matmul, Config G (B=48, M=12K, K=1536, N=1536, FP32)"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import triton
import triton.language as tl

# ── Config G ──
BATCH, M, K, N = 48, 12_000, 1_536, 1_536
DTYPE = torch.float32

# ── Triton kernel (Hopper optimized) ──
def get_autotune_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ]

@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def matmul_triton_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_aq, stride_am, stride_ak,
    stride_bq, stride_bk, stride_bn,
    stride_cq, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    a_ptr += pid_q * stride_aq
    b_ptr += pid_q * stride_bq
    c_ptr += pid_q * stride_cq

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & ((offs_k[None, :] + k * BLOCK_SIZE_K) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((offs_k[:, None] + k * BLOCK_SIZE_K) < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(a_ptr.dtype.element_ty), mask=c_mask)

def run_triton_matmul(A, B):
    B_batch, M, K = A.shape
    _, _, N = B.shape
    C = torch.empty((B_batch, M, N), device=A.device, dtype=A.dtype)
    grid = lambda META: (
        B_batch,
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_triton_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    return C

if __name__ == "__main__":
    print(f"Config G | Triton | matmul ({BATCH},{M},{K})x({BATCH},{K},{N}) | {DTYPE}")
    A = torch.randn(BATCH, M, K, device="cuda", dtype=DTYPE)
    B = torch.randn(BATCH, K, N, device="cuda", dtype=DTYPE)

    for _ in range(2):
        C = run_triton_matmul(A, B)
    torch.cuda.synchronize()

    C = run_triton_matmul(A, B)
    torch.cuda.synchronize()

    ref = torch.bmm(A, B)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-3), \
        f"FAIL: max diff = {(C - ref).abs().max().item()}"
    print("Correctness: PASS")
