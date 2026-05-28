# Kernel Profiling: CUDA vs Triton vs Helion on NVIDIA H200

Comparative GPU kernel implementations and Nsight Compute profiling artifacts for **CUDA**, **Triton**, and **Helion** — benchmarked on an NVIDIA H200 (Hopper architecture, 141 GB HBM3e).

> **Blog post:** _Coming soon_ <!-- TODO: replace with link -->

## Motivation

Hand-writing CUDA is the traditional path to high-performance GPU kernels, but modern alternatives like Triton and Helion promise comparable throughput with dramatically less code. This repository provides the full source and raw profiling data behind a systematic comparison across three workload types at multiple scales and precisions.

## Operations

| Operation | Directory | Description |
|-----------|-----------|-------------|
| Vector Addition | [`vec-add/`](vec-add/) | Element-wise add on contiguous 1-D buffers |
| Batched MatMul | [`matmul/`](matmul/) | Batched GEMM: `(B, M, K) @ (B, K, N)` |
| Softmax | [`softmax/`](softmax/) | Online softmax forward + backward along hidden dim |

Each operation is implemented in all three frameworks with identical APIs, making them drop-in replacements for profiling comparison.

## Configuration Profiles

All operations share three workload configurations derived from transformer model parameters:

| Config | Batch | Seq / M | Hidden / K, N | Dtype | Intent |
|--------|-------|---------|---------------|-------|--------|
| **configG** | 48 | 12,000 | 1,536 | FP32 | General — moderate compute + memory |
| **configH** | 24 | 48,000 | 2,048 | FP32 | Heavy — stress memory capacity + bandwidth |
| **mixedMP** | 64 | 16,000 | 2,048 | FP16 | Mixed precision — FP16 I/O, FP32 accumulation |

### Mapped to each operation

| Operation | Config G | Config H | Mixed MP |
|-----------|----------|----------|----------|
| **vec-add** | N = 884 M | N = 2.36 B | N = 2.10 B (FP16) |
| **matmul** | (48, 12K, 1536) × (48, 1536, 1536) | (24, 48K, 2048) × (24, 2048, 2048) | (64, 16K, 2048) × (64, 2048, 2048) FP16 |
| **softmax** | (B, S, H) = (48, 12K, 1536) | (24, 48K, 2048) | (64, 16K, 2048) FP16 |

## Repository Structure

```
├── vec-add/
│   ├── cuda/
│   │   ├── kernel/          # Python launchers with inline CUDA
│   │   └── results/         # .ncu-rep profiling reports
│   ├── triton/
│   │   ├── kernel/          # @triton.jit kernels
│   │   └── results/
│   └── helion/
│       ├── kernel/          # @helion.kernel DSL
│       └── results/
├── matmul/
│   └── (same layout)
├── softmax/
│   └── (same layout)
├── archive/                  # Early-iteration artifacts (old naming)
├── Kernel_Profiling_Research_Report.pdf
├── requirements.txt
└── LICENSE
```

## Naming Convention

```
<op>_<framework>_<config>.<ext>
```

| Token | Values |
|-------|--------|
| `<op>` | `vadd`, `matmul`, `softmax_fwd_bwd` |
| `<framework>` | `cuda`, `triton`, `helion` |
| `<config>` | `configG`, `configH`, `mixedMP` |
| `<ext>` | `.py` (kernel source), `.ncu-rep` (Nsight Compute report) |

Kernel and result files share the same basename:
```
matmul_triton_configG.py  →  matmul_triton_configG.ncu-rep
```

## Getting Started

### Prerequisites

- NVIDIA GPU (tested on H200, Hopper architecture)
- CUDA 12.8+ toolkit
- Python 3.10+

### Installation

```bash
git clone https://github.com/XAheli/Kernel-Profiling.git
cd Kernel-Profiling
pip install -r requirements.txt
```

### Running a Kernel

Each kernel file is a standalone script that allocates tensors, runs the kernel with warmup, and checks correctness against PyTorch:

```bash
python matmul/triton/kernel/matmul_triton_configG.py
```

### Profiling with Nsight Compute

```bash
ncu --set full -o matmul/triton/results/matmul_triton_configG \
    python matmul/triton/kernel/matmul_triton_configG.py
```

### Viewing Results

Open `.ncu-rep` files in [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute):

```bash
ncu-ui matmul/triton/results/matmul_triton_configG.ncu-rep
```

## Key Findings

Detailed analysis with metrics tables lives in the [Kernel Profiling Research Report](Kernel_Profiling_Research_Report.pdf). Summary:

- **Vec-add (memory-bound):** Triton and Helion reach 86–92% DRAM bandwidth utilization vs 36–68% for naive CUDA.
- **MatMul (compute-bound):** Naive CUDA is 10–100× slower due to lack of tensor core usage; Triton/Helion autotune to near-peak throughput.
- **Softmax:** Forward pass achieves up to 93.5% compute throughput (Triton FP16); backward pass is memory-bound with high stall rates due to reductions.
- **Mixed precision:** FP16/BF16 yields 10–30× speedups on Triton/Helion via tensor cores.

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA H200 (Hopper) |
| Memory | 141 GB HBM3e |
| CUDA Toolkit | 12.8 |
| Driver | 570.x |

## Profiling Tools

| Tool | Purpose | Output |
|------|---------|--------|
| **Nsight Compute** (`ncu`) | Kernel-level GPU metrics | `.ncu-rep` |
| **Nsight Systems** (`nsys`) | System-level timeline | `.nsys-rep` |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{poddar2026kernelprofiling,
  author       = {Poddar, Aheli},
  title        = {Kernel Profiling: Comparative Analysis of CUDA, Triton, and Helion GPU Kernels},
  year         = {2026},
  url          = {https://github.com/XAheli/Kernel-Profiling}
}
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
