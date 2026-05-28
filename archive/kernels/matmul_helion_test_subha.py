#!/usr/bin/env python3
"""Helion batched matmul: (B, M, K) x (B, K, N) -> (B, M, N)"""
import argparse
import torch
import helion
import helion.language as hl


# ------------------------------------------------------------------
# Helion kernel (Batched Matmul)
# (B, M, K) x (B, K, N) -> (B, M, N)
# ------------------------------------------------------------------
@helion.kernel()
def batched_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    B, M, K = x.size()
    _, K2, N = y.size()

    assert K == K2

    out = torch.empty([B, M, N], dtype=x.dtype, device=x.device)

    for tile_b, tile_m, tile_n in hl.tile([B, M, N]):

        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(K):

            a = x[tile_b, tile_m, tile_k]
            b = y[tile_b, tile_k, tile_n]

            acc = acc + (a @ b)

        out[tile_b, tile_m, tile_n] = acc

    return out


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def run_kernel(batch, seq, hidden, dtype, mode):

    B = batch
    M = seq
    K = hidden
    N = hidden

    print(f"\nConfig: B={B} M={M} K={K} N={N}")
    print(f"Dtype: {dtype}")
    print(f"Mode: {mode}\n")

    torch.manual_seed(0)

    x = torch.randn(B, M, K, dtype=dtype, device="cuda")
    y = torch.randn(B, K, N, dtype=dtype, device="cuda")

    # --------------------------------------------------------------
    # Cache / warmup
    # --------------------------------------------------------------
    out = batched_matmul_kernel(x, y)
    torch.cuda.synchronize()

    ref = torch.bmm(x, y)

    assert torch.allclose(out, ref, atol=1e-3), "Mismatch!"

    # --------------------------------------------------------------
    # Profiling launch
    # --------------------------------------------------------------
    if mode == "profile":

        torch.cuda.nvtx.range_push("batched_matmul_profile")

        out = batched_matmul_kernel(x, y)

        torch.cuda.synchronize()

        torch.cuda.nvtx.range_pop()

    print("Execution OK")


# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)

    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp32"
    )

    parser.add_argument(
        "--mode",
        choices=["cache", "profile"],
        required=True,
    )

    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    run_kernel(
        batch=args.batch,
        seq=args.seq,
        hidden=args.hidden,
        dtype=dtype,
        mode=args.mode,
    )
