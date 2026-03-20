"""
Standalone Helion batched matmul runner.

Usage:
  # Step 1 – Warmup / compile & cache the autotuned config (no NCU):
  python bmm_helion.py --batch 64 --m 16384 --n 1024 --k 1024 --warmup-only
  
  # Step 2 – Profile the cached kernel under NCU:
  ncu --set full --profile-from-start off -o ncu_helion_ConfigC \
      python bmm_helion.py --batch 64 --m 16384 --n 1024 --k 1024
"""

import torch
import argparse
import helion
import helion.language as hl


# ==========================================
# Helion BMM Kernel
# ==========================================
@helion.kernel(static_shapes=True)
def run_helion_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty([b, m, n], device=A.device, dtype=A.dtype)

    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n])
        out[tile_b, tile_m, tile_n] = acc.to(A.dtype)
    return out


# ==========================================
# Warmup / Compile
# ==========================================
def warmup_kernel(A, B):
    """Run kernel to trigger autotuning and cache the best config.
    Call this WITHOUT NCU so autotuning runs at full speed."""
    print("  Warmup: autotuning + caching Helion BMM...")
    for _ in range(3):
        _ = run_helion_bmm(A, B)
    torch.cuda.synchronize()
    print("  Warmup complete. Best config is now cached.")


# ==========================================
# Profiled run
# ==========================================
def profile_kernel(A, B):
    # One warm run with the CACHED config (no autotuning)
    print("  Re-running cached kernel once to warm caches...")
    _ = run_helion_bmm(A, B)
    torch.cuda.synchronize()

    # Signal NCU to START collecting (requires --profile-from-start off)
    print("  Starting profiled run...")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("profile_helion_bmm")

    result = run_helion_bmm(A, B)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    print("  Profiled run complete.")
    return result


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helion BMM profiler")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--fp16", action="store_true", help="Use float16 instead of float32")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Only run warmup/autotuning (no profiling). "
                             "Pre-cache configs before running under NCU.")

    args = parser.parse_args()
    dtype = torch.float16 if args.fp16 else torch.float32

    A = torch.randn(args.batch, args.m, args.k, device="cuda", dtype=dtype)
    B = torch.randn(args.batch, args.k, args.n, device="cuda", dtype=dtype)

    if args.warmup_only:
        warmup_kernel(A, B)
    else:
        profile_kernel(A, B)
