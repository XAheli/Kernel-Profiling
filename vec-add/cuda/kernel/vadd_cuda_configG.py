#!/usr/bin/env python3
"""NCU profiling: CUDA vector-add, Config G (B=48, S=12K, H=1536, FP32)"""
import os

_tmp_dir = os.path.expanduser("~/triton-profiling-tmp")
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")

import torch
from torch.utils.cpp_extension import load_inline

# ── Config G ──
BATCH, SEQ, HIDDEN = 48, 12_000, 1_536
DTYPE = torch.float32
N = BATCH * SEQ * HIDDEN

# ── CUDA kernel ──
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void vector_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
          scalar_t* __restrict__ c,
    int64_t n
) {
    int64_t idx    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    for (; idx < n; idx += stride)
        c[idx] = a[idx] + b[idx];
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),  "inputs must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous(),            "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(),            "b must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(),       "shape mismatch");
    auto    c       = torch::empty_like(a);
    int64_t n       = a.numel();
    int     threads = 256;
    int     blocks  = (int)std::min((n + threads - 1) / threads, (int64_t)65535);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "vector_add_cuda", ([&] {
        vector_add_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(), n);
    }));
    return c;
}
"""
CPP_SRC = "torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);"

build_dir = os.path.join(_tmp_dir, "vector_add_cuda_ext")
os.makedirs(build_dir, exist_ok=True)
_ext = load_inline(
    name="vector_add_cuda_ext",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["vector_add_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    build_directory=build_dir,
    verbose=False,
)

def cuda_vadd(a, b):
    return _ext.vector_add_cuda(a, b)

if __name__ == "__main__":
    print(f"Config G | CUDA | N={N:,} | {DTYPE}")
    a = torch.ones(N, device="cuda", dtype=DTYPE)
    b = torch.full((N,), 2.0, device="cuda", dtype=DTYPE)

    for _ in range(2):
        c = cuda_vadd(a, b)
    torch.cuda.synchronize()

    c = cuda_vadd(a, b)
    torch.cuda.synchronize()

    assert torch.allclose(c, torch.full_like(c, 3.0), atol=1e-5), "FAIL"
    print("Correctness: PASS")
