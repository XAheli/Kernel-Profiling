#!/usr/bin/env python3
"""Fused 3D Softmax forward + backward – Helion implementation

Helion kernel for softmax along dim=-1 on (Batch, Seq, Hidden) tensors.

Design decisions (from official Helion softmax examples):
- Reshape 3D (B, S, H) -> 2D (B*S, H) — Helion softmax kernels operate on 2D.
- Two-pass online softmax (forward): fuses max and sum into one pass using the
  correction term exp(m_old - m_new). Tiles BOTH row and col dimensions.
  Uses hl.register_block_size for independent tiling control.
- dim=1 not dim=-1: Helion code generation has issues with negative dim indexing.
- No keepdim: uses manual [:, None] broadcasting instead.
- Backward: tiles both dimensions — first loop computes sum(y * dy) per row,
  second loop computes dx = y * (dy - sum_per_row).
"""
import os
os.makedirs(os.path.expanduser("~/triton-profiling-tmp"), exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/triton-profiling-tmp")

import torch
import helion
import helion.language as hl


@helion.kernel()
def _helion_softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    """Two-pass online softmax: fuses max+sum in pass 1, normalizes in pass 2."""
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        # Running max and denominator (online softmax)
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        # Pass 1: compute max and sum(exp) in one pass
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            # Correct previous sum with new max, add current tile's contribution
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        # Pass 2: normalize
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@helion.kernel()
def _helion_softmax_bwd(
    grad_output: torch.Tensor, softmax_output: torch.Tensor
) -> torch.Tensor:
    """Backward: dx = y * (dy - sum(y * dy, dim=1))"""
    m, n = grad_output.size()
    grad_input = torch.empty_like(grad_output)
    for tile_m in hl.tile(m):
        # Pass 1: compute sum(y * dy) per row
        sum_per_row = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            sum_per_row += torch.sum(
                softmax_output[tile_m, tile_n] * grad_output[tile_m, tile_n], dim=1
            )
        # Pass 2: dx = y * (dy - sum_per_row)
        for tile_n in hl.tile(n):
            grad_input[tile_m, tile_n] = softmax_output[tile_m, tile_n] * (
                grad_output[tile_m, tile_n] - sum_per_row[:, None]
            )
    return grad_input


def helion_softmax_fwd(x):
    """Forward: (B, S, H) -> softmax along dim=-1"""
    assert x.is_contiguous() and x.dim() == 3
    B, S, H = x.shape
    out_2d = _helion_softmax_fwd(x.view(B * S, H))
    return out_2d.view(B, S, H)


def helion_softmax_bwd(grad_output, output):
    """Backward: dx = y * (dy - sum(y * dy))"""
    assert grad_output.is_contiguous() and output.is_contiguous()
    B, S, H = output.shape
    dx_2d = _helion_softmax_bwd(grad_output.view(B * S, H), output.view(B * S, H))
    return dx_2d.view(B, S, H)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Helion Softmax fwd+bwd")
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

    print(f"Helion Softmax fwd+bwd | ({B}, {S}, {H}) | {dt}")
    x = torch.randn(B, S, H, device="cuda", dtype=dt)

    # Warmup (JIT compilation + autotune on first call)
    for _ in range(2):
        y = helion_softmax_fwd(x)
    torch.cuda.synchronize()

    # Forward
    y = helion_softmax_fwd(x)
    torch.cuda.synchronize()
    ref = torch.softmax(x.float(), dim=-1).to(dt)
    assert torch.allclose(y.float(), ref.float(), atol=atol, rtol=1e-3), \
        f"FWD FAIL: max diff = {(y.float() - ref.float()).abs().max().item()}"
    print("Forward correctness: PASS")

    # Backward
    dy = torch.randn_like(y)
    dx = helion_softmax_bwd(dy, y)
    torch.cuda.synchronize()
    ref_dx = y.float() * (dy.float() - (y.float() * dy.float()).sum(dim=-1, keepdim=True))
    assert torch.allclose(dx.float(), ref_dx.float(), atol=atol, rtol=1e-3), \
        f"BWD FAIL: max diff = {(dx.float() - ref_dx.float()).abs().max().item()}"
    print("Backward correctness: PASS")
