# Tiled SIMD Matrix Engine

A high-performance C++ Matrix Multiplication (GEMM) and Vector (GEMV) engine optimized for modern CPU architectures. This project demonstrates significant speedups through **AVX2 SIMD Intrinsics**, **Cache-aware Tiling**, and **Multi-threaded 2D Grid Partitioning**.


## 🔹 Matrix-Vector Multiplication (GEMV)

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

## 🔸 Matrix-Matrix Multiplication (GEMM)

The current evolution stage. Unlike GEMV, GEMM is heavily compute-bound and requires aggressive cache reuse and vectorization.

### GEMM Performance:
*Benchmark conducted on a Deep Learning FC Layer Profile (M=250, K=1000, N=4000).*

| Optimization Level | Strategy | Threads | Time (ms) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Naive Triple Loop | 1 | 2508.76 | 1.00x |
| **Memory Opt** | Tiled + Packed Transpose | 1 | 849.80 | 2.95x |
| **Vectorized** | SIMD + Broadcasting | 1 | 261.29 | 9.60x |
| **Parallel (Optimal)** | **2D Grid SIMD Engine** | **4** | **119.49** | **21.00x** |
| **Parallel (Saturated)** | 2D Grid SIMD Engine | 8 | 130.37 | 19.24x |

*more benchmarks at execution*

*Note: The performance plateau between 4 and 8 threads perfectly illustrates hardware saturation on a 4-core physical CPU. Hyper-threading (logical cores) provides no additional ALUs for heavy AVX2 instructions, leading to a slight regression due to context-switching and memory bandwidth contention.*

---

## 🧠 Key Technical Implementations

### 1. Vectorized Outer-Product (Broadcasting)
To leverage AVX2 registers, the engine transitions from a standard Dot Product to an **Outer-Product pattern**. 
* **The Operation:** A single scalar from Matrix $A$ is broadcasted across a 256-bit register.
* **SIMD Execution:** It is multiplied by 8 contiguous elements of Matrix $B$ using FMA (`_mm256_fmadd_ps`).
* **Efficiency:** This approach eliminates the need for transposing Matrix $B$ during the packing stage, significantly reducing memory overhead and maximizing ALU throughput.

### 2. 2D Grid Partitioning & Thread Scaling
The `ParallelExecutor` core utilizes a 2D Grid topology to distribute work, revealing key hardware boundaries:
* **Cache Reuse:** By splitting the output matrix $C$ into blocks (Macro-tiles) rather than simple row-strips, we maximize the reuse of shared **L3 Cache** for matrices $A$ and $B$ across different threads.
* **Load Balancing:** The engine dynamically calculates the most "square-like" grid for any given thread count, ensuring optimal work distribution and handling edge-case tails for unaligned dimensions.
* **The Hardware Ceiling:** Benchmarks show peak execution at 4 threads with a regression at 8 threads. This is a textbook example of **ALU saturation**. Dense AVX2/FMA operations fully occupy the physical CPU cores; therefore, logical cores cannot extract further instruction-level parallelism and only introduce memory subsystem bottlenecks.

### 3. Cache-Aware Tiling
Micro-tiling (fixed at $32 \times 32$) ensures that the active data set for any given computation fits entirely within the **L1 Data Cache**, preventing costly stalls and cache evictions during the inner-most computation loops.

---

### 🧠 Technical Deep Dive: Dot Product vs. Outer Product (Broadcasting)

This project showcases the transition between two fundamental execution patterns in GEMM optimization:

#### 1. The Dot Product Approach
In the scalar tiled version, we calculate each element $C[i][j]$ individually:
* **The Operation:** $C[i][j]=\sum(A[i][k]\times B[k][j])$.
* **The Memory Challenge:** Since $B$ is stored in Row-Major order, accessing $B[k][j]$ for a fixed $j$ while incrementing $k$ results in **vertical strides** through memory, causing severe Cache Misses.
* **The Solution:** We **transposed (packed)** the $B$ tiles into local SRAM-like buffers to ensure $B[k][j]$ elements are contiguous in memory during the dot product.

#### 2. The Outer Product & Broadcasting Approach
To leverage SIMD (AVX2), we flipped the logic to update multiple elements of $C$ at once:
* **The Operation:** One scalar $A[i][k]$ is **broadcasted** (duplicated) into a 256-bit SIMD register.
* **Vectorized Execution:** We load **8 contiguous elements** of $B$ (from the same row $k$) into another register.
* **FMA Acceleration:** We use `_mm256_fmadd_ps` to compute $A[i][k]\times B[k][j:j+7]$ and accumulate the result directly into 8 elements of the $C$ matrix.
* **The Efficiency Win:** Since we are now consuming $B$ along its rows (horizontal access), **we no longer need to transpose the B-tiles**. We simply load them as-is, saving significant packing overhead and maximizing the throughput of the Arithmetic Logic Unit (ALU).

---

## 📂 Project Structure
```text
.
├── apps/               # Benchmark drivers (GEMM_Main, GEMV_Main)
├── include/
│   ├── core/           # ParallelExecutor (The 2D Grid Engine)
│   ├── gemm/           # Strategy interfaces & SIMD Kernels
│   └── gemv/           # WeightLoading & Streaming logic
├── src/                # Implementations
└── CMakeLists.txt      # Optimized build configuration (-O3, -mavx2)
```

---

## 🚀 Getting Started

### Prerequisites
* **CPU:** x86_64 with AVX2 and FMA support.
* **Compiler:** GCC 10+ or Clang 11+ (supporting C++20).
* **Build System:** CMake 3.10+.

### Compilation
```bash
mkdir build && cd build
cmake ..
make
```

## Running Benchmarks (Execution)
```bash
# Run GEMM Performance Benchmark
./build/gemm_bench

# Run GEMV Application
./build/gemv_app
```

### Cleaning
```bash
# Standard clean
cmake --build build --target clean
# Or simply
rm -rf build/
```