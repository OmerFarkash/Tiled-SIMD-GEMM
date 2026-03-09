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

## 🔸 Phase 2 & 3: Matrix-Matrix Multiplication (GEMM)

The current evolution stage. Unlike GEMV, GEMM is heavily compute-bound and requires aggressive cache reuse and vectorization.

### 🧠 The Paradigm Shift: Dot Product vs. Outer Product
This engine implements a fundamental algorithmic shift to maximize hardware utilization:

* **Phase 2 (Dot Product Approach):** To compute a single element in the result matrix, a row of A is multiplied by a column of B. To avoid massive cache misses caused by column-wise memory access, the B matrix tile is dynamically **transposed (packed)** into the local SRAM buffer.
* **Phase 3 (Outer Product & Broadcasting):** Utilizing AVX2 `_mm256_broadcast_ps`, the engine shifts to an outer-product formulation. A single scalar from A is broadcasted into a 256-bit register and multiplied against 8 contiguous elements of B using FMA. 
  * **The Win:** Transposing the B tile is no longer required. The kernel processes 8 columns of the result matrix simultaneously, dramatically increasing Arithmetic Intensity and achieving a massive speedup.

### 📊 GEMM Performance (Deep Learning Profile)
Benchmarking an asymmetric, unaligned Fully Connected (FC) layer profile common in Deep Learning workloads.

**Dimensions:** $M=250$ (Batch), $K=1000$ (In Features), $N=4000$ (Out Features)
**Execution:** Single Thread (Baseline for parallel scaling) | **Optimal Cache Sweet-Spot:** 32x32 Tiles

| Algorithm | Execution Time | Speedup vs Naive | Notes |
| :--- | :--- | :--- | :--- |
| **Naive GEMM** | 2101.90 ms | 1.00x | Heavy Cache Misses |
| **Naive Transpose** | 1229.57 ms | 1.71x | Global DDR Transpose |
| **Tiled Packed Transpose (64)** | 842.83 ms | 2.49x | Phase 2: Cache Locality (Dot Product) |
| **Tiled SIMD + Broadcast (32)** | **300.18 ms** | **7.00x** | Phase 3: Hardware Utilization (Outer Product) |

*Validation: All optimized kernels are mathematically verified against the naive baseline with strict tolerance bounds.*

### 🧠 Technical Deep Dive: Dot Product vs. Outer Product (Broadcasting)

This project showcases the transition between two fundamental execution patterns in GEMM optimization:

#### 1. The Dot Product Approach (Phase 2)
In the scalar tiled version, we calculate each element $C[i][j]$ individually:
* **The Operation:** $C[i][j] = \sum (A[i][k] \times B[k][j])$.
* **The Memory Challenge:** Since $B$ is stored in Row-Major order, accessing $B[k][j]$ for a fixed $j$ while incrementing $k$ results in **vertical strides** through memory, causing severe Cache Misses.
* **The Solution:** We **transposed (packed)** the $B$ tiles into local SRAM-like buffers to ensure $B[k][j]$ elements are contiguous in memory during the dot product.

#### 2. The Outer Product & Broadcasting Approach (Phase 3)
To leverage SIMD (AVX2), we flipped the logic to update multiple elements of $C$ at once:
* **The Operation:** One scalar $A[i][k]$ is **broadcasted** (duplicated) into a 256-bit SIMD register.
* **Vectorized Execution:** We load **8 contiguous elements** of $B$ (from the same row $k$) into another register.
* **FMA Acceleration:** We use `_mm256_fmadd_ps` to compute $A[i][k] \times B[k][j:j+7]$ and accumulate the result directly into 8 elements of the $C$ matrix.
* **The Efficiency Win:** Since we are now consuming $B$ along its rows (horizontal access), **we no longer need to transpose the B-tiles**. We simply load them as-is, saving significant packing overhead and maximizing the throughput of the Arithmetic Logic Unit (ALU).

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