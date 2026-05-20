# Kernel Prof

> Unified repository for GPU kernel implementations and profiling results across **CUDA**, **Triton**, and **Helion**.

---

## Overview

This repo provides a standardized structure for storing GPU kernel source code alongside their profiling artifacts. Each kernel operation is organized by framework, making it easy to compare implementations and performance across CUDA, Triton, and Helion.

**Currently tracked operations:**

| Operation | Directory | Description |
|-----------|-----------|-------------|
| Vector Addition | [`vec-add/`](vec-add/) | Element-wise vector addition |
| Batch Matrix Multiplication | [`matmul/`](matmul/) | Batchwise matrix multiplication |
| Softmax | [`softmax/`](softmax/) | Softmax activation |

---

## Directory Layout

Every operation follows the same hierarchy:

```
<operation>/
├── cuda/
│   ├── kernel/    # source code (.py, .cu)
│   └── results/   # profiling artifacts (.ncu-rep, .nsys-rep)
├── triton/
│   ├── kernel/
│   └── results/
└── helion/
    ├── kernel/
    └── results/
```

---

## Naming Conventions

All filenames — both kernel source and result artifacts — follow a single consistent pattern:

```
<op>_<framework>_<config>.<ext>
```

| Field | Description | Values |
|-------|-------------|--------|
| `<op>` | Short operation name | `vadd`, `matmul`, `softmax` |
| `<framework>` | Implementation framework | `cuda`, `triton`, `helion` |
| `<config>` | Configuration or variant identifier | `configG`, `configH`, `mixedMP`, `tiled`, etc. |
| `<ext>` | File extension | `.py` for kernels, `.ncu-rep` / `.nsys-rep` for results |

### Kernel → Result mapping

The result file is the **exact same base name** as its kernel, with the profiler extension:

```
vadd_triton_configG.py        →  vadd_triton_configG.ncu-rep
matmul_cuda_tiled.py          →  matmul_cuda_tiled.ncu-rep
softmax_helion_configA.py     →  softmax_helion_configA.nsys-rep
```

### Extra parameters (optional)

When profiling the same kernel with **different input sizes or data types**, append the parameters:

```
<op>_<framework>_<config>_<params>.<ext>
```

Example: `matmul_triton_configA_M2048_N2048_K2048_fp16.ncu-rep`

Common parameter tokens:

| Token | Meaning |
|-------|---------|
| `M`, `N`, `K` | Matrix dimensions |
| `rows`, `cols` | Row / column counts |
| `BLK` | Block size |
| `fp16`, `fp32`, `bf16` | Data type |
| `WARPS` | Number of warps |
| `STAGES` | Pipeline stages |

---



#