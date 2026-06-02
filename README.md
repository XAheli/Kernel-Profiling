<p align="center">
  <h1 align="center">Kernel Profiling: CUDA vs Triton vs Helion</h1>
  <p align="center">
    <strong>A systematic comparison of GPU kernel performance across three frameworks on NVIDIA H200</strong>
  </p>
  <p align="center">
    <a href="https://github.com/XAheli/Kernel-Profiling/blob/main/Kernel_Profiling_Research_Intern%20Report.pdf"><img src="https://img.shields.io/badge/Research-Report-blue" alt="Report"></a>
    <a href="https://developer.nvidia.com/tools-overview/nsight-compute/get-started"><img src="https://img.shields.io/badge/NVIDIA-Nsight_Compute-76B900?logo=nvidia" alt="Nsight Compute"></a>
    <a href=" "><img src="https://img.shields.io/badge/Red_Hat-Blog-EE0000?logo=redhat" alt="Red Hat Blog"></a>
    <!-- <a href="#"><img src="https://img.shields.io/badge/Blog_Post-Coming_Soon-orange" alt="Blog"></a> -->
  </p>
</p>

---

This repository contains the complete source code and raw [Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) profiling data for **vector addition**, **batched matrix multiplication**, and **softmax** — each implemented identically in CUDA, Triton, and Helion. Every kernel is a standalone script you can run, profile, and compare.

Built during our internship at the **Red Hat PyTorch Team** (Bangalore, India) as part of ongoing research into [GPU kernel profiling with NVIDIA Nsight Tools](https://next.redhat.com/2025/11/19/triton-kernel-profiling-with-nvidia-nsight-tools/).

---

## Highlights

| | CUDA | Triton | Helion |
|---|------|--------|--------|
| **Vec-add** (memory-bound) | 36–68% DRAM BW | 86–92% DRAM BW | 86–92% DRAM BW |
| **MatMul** (compute-bound) | 10–100x slower (no TC) | Near-peak (autotuned) | Near-peak (autotuned) |
| **Softmax fwd** (FP16) | Moderate | 93.5% compute | Competitive |
| **Mixed precision speedup** | Minimal | 10–30x via tensor cores | 10–30x via tensor cores |

---

## Quick Start

```bash
git clone https://github.com/XAheli/Kernel-Profiling.git
cd Kernel-Profiling
pip install -r requirements.txt

# Run a kernel (correctness check against PyTorch)
python matmul/triton/kernel/matmul_triton_configG.py

# Profile it
ncu --set full -o matmul/triton/results/matmul_triton_configG \
    python matmul/triton/kernel/matmul_triton_configG.py

# View results
ncu-ui matmul/triton/results/matmul_triton_configG.ncu-rep
```

**Requirements:** Python 3.10+ | CUDA 12.8+ | NVIDIA GPU (tested on H200, 141 GB HBM3e)

---

## What's Inside

```
<operation>/
├── cuda/
│   ├── kernel/       # Python + inline CUDA C++
│   └── results/      # .ncu-rep reports
├── triton/
│   ├── kernel/       # @triton.jit kernels
│   └── results/
└── helion/
    ├── kernel/       # @helion.kernel DSL
    └── results/
```

| Operation | What it tests |
|-----------|---------------|
| [`vec-add/`](vec-add/) | Pure memory throughput — element-wise add on large 1-D buffers |
| [`matmul/`](matmul/) | Compute throughput — batched GEMM with tiling and tensor cores |
| [`softmax/`](softmax/) | Mixed workload — online 2-pass softmax (forward + backward) |

---

## Workload Configurations

Each operation runs at three scales derived from transformer model dimensions:

| Config | Batch | Seq | Hidden | Dtype | Profile |
|--------|-------|-----|--------|-------|---------|
| `configG` | 48 | 12K | 1,536 | FP32 | Moderate — balanced compute/memory |
| `configH` | 24 | 48K | 2,048 | FP32 | Heavy — memory-capacity stress test |
| `mixedMP` | 64 | 16K | 2,048 | FP16 | Mixed precision — FP16 I/O, FP32 accumulation |

File naming: `<op>_<framework>_<config>.py` with matching `.ncu-rep` results.

---

## Related Resources

- [NVIDIA Nsight Compute — Getting Started](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)
- [Triton Kernel Profiling with NVIDIA Nsight Tools — Red Hat Emerging Technologies Blog](https://next.redhat.com/2025/11/19/triton-kernel-profiling-with-nvidia-nsight-tools/)
- [Full Research Report (PDF)](https://github.com/XAheli/Kernel-Profiling/blob/main/Kernel_Profiling_Research_Intern%20Report.pdf)

---

<p align="center">
  <sub>Built with care at <strong>Red Hat</strong> · PyTorch Team · Bangalore, India</sub>
</p>
