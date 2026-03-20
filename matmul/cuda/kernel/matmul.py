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
        # Hopper WGMMA/TMA friendly configs: Larger tiles, more stages
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        # Fallback for smaller/edge cases
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
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

def run_triton_bmm(A, B):
    B_batch, M, K = A.shape
    _, _, N = B.shape
    C = torch.empty((B_batch, M, N), device=A.device, dtype=A.dtype)

    # Lambda grid allows the autotuner to dynamically inject the chosen META configuration sizes
    grid = lambda META: (
        B_batch,
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )

    bmm_triton_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
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
def profile_kernel(A, B, kernel_name, warmup_runs=3):
    dispatch = {"cuda": cuda_module.run_cuda_bmm, "triton": run_triton_bmm, "helion": run_helion_bmm}
    target_func = dispatch[kernel_name]

    # Warmup phase allows Triton/Helion to run autotuning with profiling OFF
    if warmup_runs > 0:
        print(f"  Warmup (autotuning + cache, profiler is OFF)...")
        for _ in range(warmup_runs):
            _ = target_func(A, B)
        torch.cuda.synchronize()
        print(f"  Warmup complete. Starting profiled run...")

    # NVTX range for NCU/nsys; do not use cudaProfilerStart/Stop with NCU (causes "No kernels profiled")
    torch.cuda.nvtx.range_push(f"profile_{kernel_name}")

    result = target_func(A, B)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_pop()

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
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before profiled run (0 = single launch only)")

    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32

    A = torch.randn(args.batch, args.m, args.k, device="cuda", dtype=dtype)
    B = torch.randn(args.batch, args.k, args.n, device="cuda", dtype=dtype)

    profile_kernel(A, B, kernel_name=args.kernel, warmup_runs=args.warmup)