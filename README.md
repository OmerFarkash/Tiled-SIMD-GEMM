# Tiled-SIMD-Matrix-Engine 🚀

A high-performance linear algebra acceleration library in C++20. This project demonstrates the evolution of optimizing matrix operations, moving from **Matrix-Vector (GEMV)** to **Matrix-Matrix (GEMM)** multiplication by bypassing the "memory wall" using SIMD, tiling, and software-managed memory.

---

## 🏗 System Architecture

The project implements **Software-Managed Memory** concepts:
1. **DDR (Main Memory):** Large, high-latency storage.
2. **SRAM (Local Buffers):** Fast, tiled buffers simulating on-chip memory.
3. **Compute Kernels:** Hand-optimized AVX2/FMA routines.
4. **Execution Engine:** A lock-free `ParallelExecutor` for multi-threaded scaling.

---

## 🔹 Phase 1: Matrix-Vector Multiplication (GEMV)

Focuses on real-time streaming of weights using asynchronous double-buffering.

### GEMV Features:
* **Double Buffering:** Overlaps I/O (loading tiles) with computation.
* **SIMD & FMA:** Processes 8 floats per cycle using `AVX2`.
* **Robust Tail Handling:** Supports any matrix dimension without padding.

### GEMV Performance:
**Matrix Size:** $10013 \times 12452$ | **CPU Cores:** 4

| Implementation | Execution Time | Speedup |
| :--- | :--- | :--- |
| Naive (1 Thread) | 188.86 ms | 1.00x |
| SIMD (1 Thread) | 93.17 ms | ~2.03x |
| SIMD (4 Threads) | 54.40 ms | **~3.47x** |

---

## 🔸 Phase 2: Matrix-Matrix Multiplication (GEMM)

The current evolution stage. Unlike GEMV, GEMM is compute-bound and requires aggressive cache reuse.

### GEMM Strategy: Tiling & Packing
* **L1-Friendly Tiling:** Matrices are processed in $64 \times 64$ blocks to stay within the fastest cache layers.
* **On-the-fly Packing:** Transposing the "B" matrix tiles into a contiguous "SRAM" buffer during the load phase to ensure purely linear memory access.
* **Spatial Locality:** Maximizing cache hits by aligning data access with the hardware prefetcher.

### GEMM Performance:
**Matrix Size:** $1025 \times 1025$ | **Single Thread (Baseline)**

| Algorithm | Execution Time | Speedup | Notes |
| :--- | :--- | :--- | :--- |
| **Naive GEMM** | 2639.66 ms | 1.00x | Heavy Cache Misses |
| **Naive Transpose** | 1265.07 ms | 2.09x | Global DDR Transpose |
| **Tiled Packed** | 759.68 ms | **3.47x** | Optimized Cache Reuse |

---

## 🚀 Getting Started

### Compilation
```bash
mkdir build && cd build
cmake ..
make
```

## Running Benchmarks
* GEMV: `./gemv_app`

* GEMM: `./gemm_bench`


### 📂 Roadmap
[x] Phase 1: Optimized GEMV with Double Buffering.

[x] Phase 2a: Tiled GEMM with On-the-fly Packing (Tail handling included).

[ ] Phase 2b: SIMD Kernel Sharpening (AVX2/FMA & Broadcasting).

[ ] Phase 2c: Multi-threaded GEMM Grid Execution.