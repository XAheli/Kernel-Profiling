# Kernel Profiling: CUDA vs Triton vs Helion

**A systematic comparison of GPU kernel performance across three frameworks on NVIDIA H200.**

This repository contains the complete source code and raw Nsight Compute profiling data for vector addition, batched matrix multiplication, and softmax — each implemented identically in CUDA, Triton, and Helion. Every kernel is a standalone script you can run, profile, and compare.

> **Blog post:** _Coming soon_ <!-- TODO: replace with link -->
>
> **Full report:** [Kernel Profiling Research Report (PDF)](Kernel_Profiling_Research_Report.pdf)

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

**Requirements:** Python 3.10+, CUDA 12.8+, NVIDIA GPU (tested on H200 141 GB HBM3e)

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

Three operations, three frameworks, three workload scales:

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

File naming follows: `<op>_<framework>_<config>.py` with matching `.ncu-rep` results.

---

## Citation

```bibtex
@misc{poddar2026kernelprofiling,
  author       = {Poddar, Aheli},
  title        = {Kernel Profiling: Comparative Analysis of CUDA, Triton, and Helion GPU Kernels},
  year         = {2026},
  url          = {https://github.com/XAheli/Kernel-Profiling}
}
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).
