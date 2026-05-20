#!/usr/bin/env python3
"""Fused 3D Softmax forward + backward – CUDA implementation (2-pass online)

CUDA kernel for softmax along dim=-1 on (Batch, Seq, Hidden) tensors.

Design decisions:
- 1 block per row: threads cooperate on reduction within a row (ref2 kernel 3)
- Warp shuffle __shfl_down_sync for intra-warp reduction: register-speed,
  eliminates __syncthreads within warp (ref2 kernel 4)
- 2-pass online softmax (forward): each thread maintains running (max, norm)
  with exp(m_old - m_new) correction. Block-level online reduction combines
  per-thread pairs. Pass 2 normalizes. 2 reads + 1 write per element.
- 2-pass backward: sum(y*dy) reduction, then dx = y*(dy - dot).
- FP32 accumulation: numerical stability for FP16/BF16 inputs
- Thread count: min(1024, round_up_to_warp(hidden)) — maximize parallelism
"""
import os

_tmp_dir = os.path.expanduser("~/triton-profiling-tmp")
os.makedirs(_tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = _tmp_dir
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// ──────────────────────────────────────────────────────────────
// Warp-level online reduction: combines (max, norm) pairs using
// the correction factor exp(m_old - m_new) via __shfl_down_sync.
// ──────────────────────────────────────────────────────────────

__device__ __forceinline__ void warp_reduce_online(float& mx, float& dn) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float o_mx = __shfl_down_sync(0xFFFFFFFF, mx, offset);
        float o_dn = __shfl_down_sync(0xFFFFFFFF, dn, offset);
        float new_mx = fmaxf(mx, o_mx);
        // Guard: when both are -inf, (-inf)-(-inf)=NaN would poison the result
        dn = (new_mx > -INFINITY)
            ? dn * expf(mx - new_mx) + o_dn * expf(o_mx - new_mx)
            : 0.0f;
        mx = new_mx;
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ──────────────────────────────────────────────────────────────
// Block-level reductions.
// block_reduce_online: warp shuffle + smem for (max, norm) pairs.
//   smem layout: [0..nw-1] = max, [nw..2*nw-1] = norm.
// block_reduce_sum: unchanged, used by backward kernel.
// ──────────────────────────────────────────────────────────────

__device__ void block_reduce_online(float& mx, float& dn, float* smem) {
    int wid = threadIdx.x / warpSize;
    int lid = threadIdx.x % warpSize;
    int nw  = (blockDim.x + warpSize - 1) / warpSize;

    warp_reduce_online(mx, dn);
    if (lid == 0) { smem[wid] = mx; smem[nw + wid] = dn; }
    __syncthreads();

    if (wid == 0) {
        mx = (lid < nw) ? smem[lid] : -INFINITY;
        dn = (lid < nw) ? smem[nw + lid] : 0.0f;
        warp_reduce_online(mx, dn);
        if (lid == 0) { smem[0] = mx; smem[nw] = dn; }
    }
    __syncthreads();
    mx = smem[0];
    dn = smem[nw];
}

__device__ float block_reduce_sum(float val, float* smem) {
    int wid = threadIdx.x / warpSize;
    int lid = threadIdx.x % warpSize;
    int nw  = (blockDim.x + warpSize - 1) / warpSize;

    val = warp_reduce_sum(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lid < nw) ? smem[lid] : 0.0f;
        val = warp_reduce_sum(val);
        if (lid == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

// ──────────────────────────────────────────────────────────────
// Forward kernel: 2-pass online softmax, 1 block per row.
// Pass 1: each thread maintains running (max, norm) with correction
//         exp(m_old - m_new), then block_reduce_online combines pairs.
// Pass 2: normalize using converged max and denominator.
// ──────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void softmax_fwd_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const scalar_t* x = input  + (int64_t)row * cols;
    scalar_t*       y = output + (int64_t)row * cols;

    extern __shared__ float smem[];

    float mx = -INFINITY;
    float dn = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        float new_mx = fmaxf(mx, val);
        dn = dn * expf(mx - new_mx) + expf(val - new_mx);
        mx = new_mx;
    }
    block_reduce_online(mx, dn, smem);

    float inv_dn = 1.0f / dn;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] = static_cast<scalar_t>(expf(static_cast<float>(x[i]) - mx) * inv_dn);
}

// ──────────────────────────────────────────────────────────────
// Backward kernel: dx = y * (dy - sum(y * dy))
// 2 passes: (1) dot = sum(y * dy), (2) dx = y * (dy - dot)
// ──────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void softmax_bwd_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ output,
    scalar_t* __restrict__ grad_input,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const scalar_t* dy = grad_output + (int64_t)row * cols;
    const scalar_t* yy = output     + (int64_t)row * cols;
    scalar_t*       dx = grad_input + (int64_t)row * cols;

    extern __shared__ float smem[];

    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        dot += static_cast<float>(yy[i]) * static_cast<float>(dy[i]);
    dot = block_reduce_sum(dot, smem);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float yi  = static_cast<float>(yy[i]);
        float dyi = static_cast<float>(dy[i]);
        dx[i] = static_cast<scalar_t>(yi * (dyi - dot));
    }
}

// ──────────────────────────────────────────────────────────────
// C++ wrappers
// ──────────────────────────────────────────────────────────────

torch::Tensor softmax_fwd_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 3,
                "input must be a contiguous 3D CUDA tensor");
    int rows = input.size(0) * input.size(1);
    int cols = input.size(2);
    auto output = torch::empty_like(input);
    int thr = std::min(1024, ((cols + 31) / 32) * 32);
    int nw = (thr + 31) / 32;
    // Two floats per warp: max and norm for online reduction
    int smem_bytes = 2 * nw * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "softmax_fwd", ([&] {
            softmax_fwd_kernel<scalar_t><<<rows, thr, smem_bytes>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                rows, cols);
        }));
    return output;
}

torch::Tensor softmax_bwd_cuda(torch::Tensor grad_output, torch::Tensor output) {
    TORCH_CHECK(grad_output.is_cuda() && output.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(grad_output.is_contiguous() && output.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");
    int rows = output.size(0) * output.size(1);
    int cols = output.size(2);
    auto grad_input = torch::empty_like(grad_output);
    int thr = std::min(1024, ((cols + 31) / 32) * 32);
    int smem_bytes = ((thr + 31) / 32) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        output.scalar_type(), "softmax_bwd", ([&] {
            softmax_bwd_kernel<scalar_t><<<rows, thr, smem_bytes>>>(
                grad_output.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), rows, cols);
        }));
    return grad_input;
}
"""

CPP_SRC = """
torch::Tensor softmax_fwd_cuda(torch::Tensor input);
torch::Tensor softmax_bwd_cuda(torch::Tensor grad_output, torch::Tensor output);
"""

build_dir = os.path.join(_tmp_dir, "softmax_cuda_ext")
os.makedirs(build_dir, exist_ok=True)
_ext = load_inline(
    name="softmax_cuda_ext",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["softmax_fwd_cuda", "softmax_bwd_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    build_directory=build_dir,
    verbose=False,
)


def cuda_softmax_fwd(x):
    """Forward pass: (B, S, H) -> softmax along dim=-1"""
    return _ext.softmax_fwd_cuda(x)


def cuda_softmax_bwd(grad_output, output):
    """Backward pass: dx = y * (dy - sum(y * dy))"""
    return _ext.softmax_bwd_cuda(grad_output, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CUDA Softmax fwd+bwd")
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

    print(f"CUDA Softmax fwd+bwd | ({B}, {S}, {H}) | {dt}")
    x = torch.randn(B, S, H, device="cuda", dtype=dt)

    for _ in range(2):
        y = cuda_softmax_fwd(x)
    torch.cuda.synchronize()

    y = cuda_softmax_fwd(x)
    torch.cuda.synchronize()
    ref = torch.softmax(x.float(), dim=-1).to(dt)
    assert torch.allclose(y.float(), ref.float(), atol=atol, rtol=1e-3), \
        f"FWD FAIL: max diff = {(y.float() - ref.float()).abs().max().item()}"
    print("Forward correctness: PASS")

    dy = torch.randn_like(y)
    dx = cuda_softmax_bwd(dy, y)
    torch.cuda.synchronize()
    ref_dx = y.float() * (dy.float() - (y.float() * dy.float()).sum(dim=-1, keepdim=True))
    assert torch.allclose(dx.float(), ref_dx.float(), atol=atol, rtol=1e-3), \
        f"BWD FAIL: max diff = {(dx.float() - ref_dx.float()).abs().max().item()}"
    print("Backward correctness: PASS")
