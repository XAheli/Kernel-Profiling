#!/usr/bin/env python3
"""NCU profiling: CUDA batched matmul, Config G (B=48, M=12K, K=1536, N=1536, FP32)"""
import os

_tmp_dir = os.path.expanduser("~/triton-profiling-tmp")
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")

import torch
from torch.utils.cpp_extension import load_inline

# ── Config G ──
BATCH, M, K, N = 48, 12_000, 1_536, 1_536
DTYPE = torch.float32

# ── CUDA kernel ──
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

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = static_cast<float>(batch_A[row * K + t * TILE_SIZE + threadIdx.x]);
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < K && col < N)
            sB[threadIdx.y][threadIdx.x] = static_cast<float>(batch_B[(t * TILE_SIZE + threadIdx.y) * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        batch_C[row * N + col] = static_cast<scalar_t>(sum);
}

torch::Tensor run_cuda_matmul(torch::Tensor A, torch::Tensor B) {
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
    name="matmul_cuda_ext",
    cpp_sources="torch::Tensor run_cuda_matmul(torch::Tensor A, torch::Tensor B);",
    cuda_sources=cuda_source,
    functions=["run_cuda_matmul"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)

if __name__ == "__main__":
    print(f"Config G | CUDA | matmul ({BATCH},{M},{K})x({BATCH},{K},{N}) | {DTYPE}")
    A = torch.randn(BATCH, M, K, device="cuda", dtype=DTYPE)
    B = torch.randn(BATCH, K, N, device="cuda", dtype=DTYPE)

    for _ in range(2):
        C = cuda_module.run_cuda_matmul(A, B)
    torch.cuda.synchronize()

    C = cuda_module.run_cuda_matmul(A, B)
    torch.cuda.synchronize()

    ref = torch.bmm(A, B)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-3), \
        f"FAIL: max diff = {(C - ref).abs().max().item()}"
    print("Correctness: PASS")
