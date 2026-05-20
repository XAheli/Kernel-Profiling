#!/usr/bin/env python3
"""Fused 3D Softmax forward + backward – Triton implementation (2-pass online)

Triton kernel for softmax along dim=-1 on (Batch, Seq, Hidden) tensors.

Design decisions:
- 2-pass online softmax (forward): BLOCK_SIZE = next_power_of_2(hidden).
  When H <= BLOCK_SIZE the loop runs once (single-tile, no correction overhead).
  For H > BLOCK_SIZE, tiles with exp(m_old - m_new) correction.
  Pass 1 fuses max + sum(exp), Pass 2 normalizes.
- 2-pass backward: Pass 1 computes sum(y * dy) across tiles, Pass 2 computes
  dx = y * (dy - dot). Same tiling as forward.
- FP32 accumulation: .to(tl.float32) on loads for numerical stability in exp/sum.
- num_warps heuristic: 4 (default), 8 (>=2048), 16 (>=8192).
- int64 indexing: row_start = row_idx * stride can exceed int32 for large tensors.
- Mask for padding: when BLOCK_SIZE > n_cols, mask prevents OOB access.
"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_fwd_kernel(
    input_ptr, output_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start = row_idx * stride_row

    mi = -float('inf')
    di = 0.0
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = (start + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf')).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        mi_new = tl.maximum(mi, tile_max)
        di = di * tl.exp(mi - mi_new) + tl.sum(tl.exp(x - mi_new), axis=0)
        mi = mi_new

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = (start + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
        y = tl.exp(x.to(tl.float32) - mi) / di
        tl.store(output_ptr + row_start + cols, y.to(x.dtype), mask=mask)


@triton.jit
def softmax_bwd_kernel(
    grad_out_ptr, output_ptr, grad_in_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start = row_idx * stride_row

    dot = 0.0
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = (start + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = cols < n_cols
        dy = tl.load(grad_out_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        y  = tl.load(output_ptr   + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(y * dy, axis=0)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = (start + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = cols < n_cols
        dy = tl.load(grad_out_ptr + row_start + cols, mask=mask, other=0.0)
        y  = tl.load(output_ptr   + row_start + cols, mask=mask, other=0.0)
        dx = y.to(tl.float32) * (dy.to(tl.float32) - dot)
        tl.store(grad_in_ptr + row_start + cols, dx.to(dy.dtype), mask=mask)


def _num_warps(block_size):
    if block_size >= 8192:
        return 16
    elif block_size >= 2048:
        return 8
    return 4


def triton_softmax_fwd(x):
    """Forward: (B, S, H) -> softmax along dim=-1 (2-pass online)"""
    assert x.is_contiguous() and x.dim() == 3
    B, S, H = x.shape
    out = torch.empty_like(x)
    n_rows = B * S
    BLOCK_SIZE = triton.next_power_of_2(H)
    softmax_fwd_kernel[(n_rows,)](
        x, out, H, H,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=_num_warps(BLOCK_SIZE),
    )
    return out


def triton_softmax_bwd(grad_output, output):
    """Backward: dx = y * (dy - sum(y * dy)) (2-pass tiled)"""
    assert grad_output.is_contiguous() and output.is_contiguous()
    B, S, H = output.shape
    grad_input = torch.empty_like(grad_output)
    n_rows = B * S
    BLOCK_SIZE = triton.next_power_of_2(H)
    softmax_bwd_kernel[(n_rows,)](
        grad_output, output, grad_input, H, H,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=_num_warps(BLOCK_SIZE),
    )
    return grad_input


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Triton Softmax fwd+bwd")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    B, S, H = args.batch, args.seq, args.hidden
    dt = dtype_map[args.dtype]
    atol = 1e-5 if dt == torch.float32 else 1e-2

    print(f"Triton Softmax fwd+bwd | ({B}, {S}, {H}) | {dt}")
    x = torch.randn(B, S, H, device="cuda", dtype=dt)

    for _ in range(2):
        y = triton_softmax_fwd(x)
    torch.cuda.synchronize()

    y = triton_softmax_fwd(x)
    torch.cuda.synchronize()
    ref = torch.softmax(x.float(), dim=-1).to(dt)
    assert torch.allclose(y.float(), ref.float(), atol=atol, rtol=1e-3), \
        f"FWD FAIL: max diff = {(y.float() - ref.float()).abs().max().item()}"
    print("Forward correctness: PASS")

    dy = torch.randn_like(y)
    dx = triton_softmax_bwd(dy, y)
    torch.cuda.synchronize()
    ref_dx = y.float() * (dy.float() - (y.float() * dy.float()).sum(dim=-1, keepdim=True))
    assert torch.allclose(dx.float(), ref_dx.float(), atol=atol, rtol=1e-3), \
        f"BWD FAIL: max diff = {(dx.float() - ref_dx.float()).abs().max().item()}"
    print("Backward correctness: PASS")
