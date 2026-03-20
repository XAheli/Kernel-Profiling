import torch
import triton
import triton.language as tl
import helion
import helion.language as hl
import argparse
from torch.utils.cpp_extension import load_inline

# ==========================================
# 1. CUDA Implementation (Supports FP32 & FP16)
# ==========================================
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

template <typename scalar_t>
__global__ void batched_matmul_kernel(
    const scalar_t* __restrict__ A, 
    const scalar_t* __restrict__ B, 
    scalar_t* __restrict__ C, 
    int batch_size, int M, int N, int K) {
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const scalar_t* batch_A = A + batch_idx * M * K;
    const scalar_t* batch_B = B + batch_idx * K * N;
    scalar_t* batch_C = C + batch_idx * M * N;

    // Use float for accumulation to mimic mixed precision and prevent overflow
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = static_cast<float>(batch_A[row * K + t * TILE_SIZE + threadIdx.x]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = static_cast<float>(batch_B[(t * TILE_SIZE + threadIdx.y) * N + col]);
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        batch_C[row * N + col] = static_cast<scalar_t>(sum);
    }
}

torch::Tensor run_cuda_bmm(torch::Tensor A, torch::Tensor B) {
    int B_batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    auto C = torch::empty({B_batch, M, N}, A.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                (M + TILE_SIZE - 1) / TILE_SIZE, 
                B_batch);
                
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "batched_matmul_kernel", ([&] {
        batched_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), 
            B_batch, M, N, K);
    }));
        
    return C;
}
"""

cuda_module = load_inline(
    name="bmm_cuda_ext",
    cpp_sources="torch::Tensor run_cuda_bmm(torch::Tensor A, torch::Tensor B);",
    cuda_sources=cuda_source,
    functions=["run_cuda_bmm"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"]
)

# ==========================================
# 2. Triton Implementation (Hopper Optimized)
# ==========================================
def get_autotune_configs():
    return [
        # --- Balanced / square-ish configs (good for Config D: M=32K, N=K=2048) ---
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        # --- Tall-skinny configs (good for Config C: M=16K, N=K=1024) ---
        # Large M-tile + small N-tile: fewer M-blocks competing for B in L2
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=5, num_warps=4),
        # --- Mixed precision / small-shape fallback ---
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_triton_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_aq, stride_am, stride_ak,
    stride_bq, stride_bk, stride_bn,
    stride_cq, stride_cm, stride_cn,
    B_batch,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Flat 1D grid → derive batch, m-tile, n-tile with L2-friendly swizzle
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tiles_per_batch = num_m_tiles * num_n_tiles

    # Batch index from flat pid
    pid_q = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch

    # GROUP_SIZE_M swizzle: cluster GROUP_SIZE_M consecutive M-tiles
    # so they share the same B columns in L2 before moving to the next N-tile
    num_pid_in_group = GROUP_SIZE_M * num_n_tiles
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    # Offset pointers to the current batch
    a_ptr += pid_q * stride_aq
    b_ptr += pid_q * stride_bq
    c_ptr += pid_q * stride_cq

    # Tile offsets
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

    # Store output tile
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(a_ptr.dtype.element_ty), mask=c_mask)

def run_triton_bmm(A, B):
    B_batch, M, K = A.shape
    _, _, N = B.shape
    C = torch.empty((B_batch, M, N), device=A.device, dtype=A.dtype)

    # Flat 1D grid: batch × m_tiles × n_tiles — swizzle happens inside the kernel
    grid = lambda META: (
        B_batch * triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    bmm_triton_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        B_batch,
    )
    return C

# ==========================================
# 3. Helion Implementation 
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
# 4. Profiling Wrapper
# ==========================================
def warmup_kernel(A, B, kernel_name):
    """Run kernel to trigger autotuning and cache the best config.
    Call this WITHOUT NCU so autotuning runs at full speed."""
    dispatch = {"cuda": cuda_module.run_cuda_bmm, "triton": run_triton_bmm, "helion": run_helion_bmm}
    target_func = dispatch[kernel_name]
    print(f"  Warmup: autotuning + caching for '{kernel_name}'...")
    for _ in range(3):
        _ = target_func(A, B)
    torch.cuda.synchronize()
    print(f"  Warmup complete. Best config is now cached.")


def profile_kernel(A, B, kernel_name):
    dispatch = {"cuda": cuda_module.run_cuda_bmm, "triton": run_triton_bmm, "helion": run_helion_bmm}
    target_func = dispatch[kernel_name]

    # Quick warmup using the CACHED config (no autotuning happens here)
    print(f"  Re-running cached kernel once to warm caches...")
    _ = target_func(A, B)
    torch.cuda.synchronize()

    # Signal NCU to START collecting (requires --profile-from-start off)
    print(f"  Starting profiled run...")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"profile_{kernel_name}")

    result = target_func(A, B)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"  Profiled run complete.")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, choices=["cuda", "triton", "helion"], required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--fp16", action="store_true", help="Use float16 instead of float32")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Only run warmup/autotuning (no profiling). "
                             "Use this to pre-cache configs before running under NCU.")

    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32

    A = torch.randn(args.batch, args.m, args.k, device="cuda", dtype=dtype)
    B = torch.randn(args.batch, args.k, args.n, device="cuda", dtype=dtype)

    if args.warmup_only:
        warmup_kernel(A, B, kernel_name=args.kernel)
    else:
        profile_kernel(A, B, kernel_name=args.kernel)