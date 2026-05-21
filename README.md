# Kernel Prof

Unified repository for GPU kernel implementations and Nsight profiling artifacts across **CUDA**, **Triton**, and **Helion** (NVIDIA H200). (Not all rep files present as size limit exceeding)

## Overview

Each operation stores source under `kernel/` and matching profiler outputs under `results/`, using the same layout for every framework so implementations and NCU/nsys captures stay easy to compare.

| Operation | Directory | Description |
|-----------|-----------|-------------|
| Vector addition | `vec-add/` | Element-wise add on contiguous 1D buffers ($n = B \cdot S \cdot H$) |
| Batch matrix multiplication | `matmul/` | Batched GEMM / BMM |
| Softmax | `softmax/` | Softmax forward and backward along the hidden dimension |

## Directory layout

```
<operation>/
├── cuda/
│   ├── kernel/     # .py launchers, inline CUDA, or .cu
│   └── results/    # .ncu-rep, .nsys-rep
├── triton/
│   ├── kernel/
│   └── results/
└── helion/
    ├── kernel/
    └── results/
```

## Naming convention

```
<op>_<framework>_<config>[_<params>].<ext>
```

| Field | Values | Example |
|-------|--------|---------|
| `<op>` | `vadd`, `matmul`, `softmax` | `vadd_cuda_configG.py` |
| `<framework>` | `cuda`, `triton`, `helion` | `softmax_triton_configA.py` |
| `<config>` | Shape / sweep id | `configA` … `configH`, `mixedMP`, `tiled` |
| `<params>` | Optional (dtype, tile size, …) | `fp16`, `M2048_N2048_K2048`, `BLK512` |
| `<ext>` | `.py` (kernel), `.ncu-rep` / `.nsys-rep` (results) | |

**Kernel → result:** same basename, profiler extension only.

```
vadd_triton_configG.py     →  vadd_triton_configG.ncu-rep
matmul_cuda_tiled.py       →  matmul_cuda_tiled.ncu-rep
softmax_helion_configA.py  →  softmax_helion_configA.nsys-rep
```

**Optional parameter tokens:** `M`, `N`, `K` (GEMM dims); `fp32`, `fp16`, `bf16`; `BLK` (block size); `WARPS`; `STAGES`.

## Configuration labels

Report sweeps use **Config A–H** and fixed dtype rows (e.g. $(64, 8\text{K}, 1024)$ in FP32/FP16/BF16). Filename `config` tokens should match those labels where possible (`configG`, `configH`, etc.).

**CUDA vecadd note:** Configs **G** and **H** use a 65535-block cap and a grid-stride loop; other configs use a standard 256-thread, one-element-per-thread launch unless noted in the kernel file.

## Profiling workflow

1. Warm up / autotune off-NCU (Triton `@autotune`, Helion JIT) where applicable.
2. Profile with Nsight Compute (`ncu --set full …`) or Nsight Systems as needed.
3. Drop the `.ncu-rep` / `.nsys-rep` next to the kernel name under `results/`.

Environment hints used in development: CUDA 12.x toolchain, `TMPDIR` for Triton/Helion cache, contiguous GPU tensors for 1D vecadd launches.

## Related work

Internship analysis and tables (duration, DRAM, occupancy, instruction mix, GFLOPs) live in the separate **Kernel Profiling Research Report**; this repo holds the reproducible kernels and raw profiler artifacts referenced there.

