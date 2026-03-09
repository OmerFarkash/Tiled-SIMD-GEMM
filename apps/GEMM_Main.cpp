#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <thread>
#include <cmath>

#include "../include/core/ParallelExecutor.hpp"
#include "../include/gemm/GEMM_Common.hpp"
#include "../include/gemm/SimdGemmStrategy.hpp"

// Function declarations from GEMM_Kernels.cpp
void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_transpose_naive(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed_dynamic(const Matrix& A, const Matrix& B, Matrix& C, int tile_size);
void gemm_tiled_simd(const Matrix& A, const Matrix& B, Matrix& C, int tile_size);

// Verification utility
bool verify(const Matrix& C1, const Matrix& C2) {
    if (C1.data.size() != C2.data.size()) return false;
    for (size_t i = 0; i < C1.data.size(); ++i) {
        if (std::abs(C1.data[i] - C2.data[i]) > 1e-3) return false;
    }
    return true;
}

int main() {
    // Deep Learning Asymmetric Profile (Unaligned)
    const int M = 250;   // Batch size
    const int K = 1000;  // Input features
    const int N = 4000;  // Output features

    std::cout << "\n=================================================================\n";
    std::cout << "       Phase 4: Full System Benchmark (SIMD + Multi-Threading)   \n";
    std::cout << "       Profile: Deep Learning FC Layer (Unaligned)               \n";
    std::cout << "       Dimensions: M=" << M << ", K=" << K << ", N=" << N << "\n";
    std::cout << "=================================================================\n\n";

    // Setup matrices
    Matrix A(M, K), B(K, N);
    Matrix C_naive(M, N), C_trans(M, N), C_packed(M, N), C_simd(M, N);

    // Deterministic initialization
    const uint32_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : A.data) val = dist(gen);
    for (auto& val : B.data) val = dist(gen);

    // -------------------------------------------------------------------------
    // PART 1: ALGORITHMIC COMPARISON (SINGLE THREAD)
    // -------------------------------------------------------------------------
    std::cout << "--- Part 1: Algorithmic Comparison (Single Thread) ---\n";
    std::cout << std::left << std::setw(32) << "Algorithm" 
              << std::setw(15) << "Time (ms)" 
              << "Speedup vs Naive" << std::endl;
    std::cout << "---------------------------------------------------------------\n";

    // 1. Naive (Baseline)
    auto start = std::chrono::high_resolution_clock::now();
    gemm_naive(A, B, C_naive);
    auto end = std::chrono::high_resolution_clock::now();
    double t_naive = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::left << std::setw(32) << "Naive GEMM" 
              << std::fixed << std::setprecision(2) << std::setw(15) << t_naive 
              << "1.00x (Baseline)" << std::endl;

    // 2. Transpose
    start = std::chrono::high_resolution_clock::now();
    gemm_transpose_naive(A, B, C_trans);
    end = std::chrono::high_resolution_clock::now();
    double t_trans = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::left << std::setw(32) << "Naive Transpose" 
              << std::setw(15) << t_trans 
              << (t_naive / t_trans) << "x" << std::endl;

    // 3. Tiled Packed (Dot-Product based)
    start = std::chrono::high_resolution_clock::now();
    gemm_tiled_packed(A, B, C_packed);
    end = std::chrono::high_resolution_clock::now();
    double t_packed = std::chrono::duration<double, std::milli>(end - start).count();
    std::string packed_name = "Tiled Packed Transpose (" + std::to_string(TILE_SIZE) + ")";
    std::cout << std::left << std::setw(32) << packed_name 
              << std::setw(15) << t_packed 
              << (t_naive / t_packed) << "x" << std::endl;

    // 4. SIMD + Broadcast (Outer-Product based)
    start = std::chrono::high_resolution_clock::now();
    gemm_tiled_simd(A, B, C_simd, 32); // Using optimal tile size
    end = std::chrono::high_resolution_clock::now();
    double t_simd = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::left << std::setw(32) << "Tiled SIMD (32)" 
              << std::setw(15) << t_simd 
              << (t_naive / t_simd) << "x" << std::endl;

    // Validation
    std::cout << "\nValidation: ";
    if (verify(C_naive, C_trans) && verify(C_naive, C_packed) && verify(C_naive, C_simd)) {
        std::cout << "[SUCCESS] All single-thread kernels produce identical results.\n\n";
    } else {
        std::cout << "[ERROR] Discrepancy found in results!\n\n";
        return 1;
    }

    // -------------------------------------------------------------------------
    // PART 2: CACHE SWEET-SPOT ANALYSIS
    // -------------------------------------------------------------------------
    std::cout << "--- Part 2: Cache Sweet-Spot Analysis (Tiled Packed) ---\n";
    std::cout << std::left << std::setw(15) << "Tile Size" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Speedup"
              << "Status" << std::endl;
    std::cout << "------------------------------------------------------\n";

    std::vector<int> tile_sizes = {16, 32, 64, 128, 256, 512};
    for (int ts : tile_sizes) {
        Matrix C_test(M, N); 
        start = std::chrono::high_resolution_clock::now();
        gemm_tiled_packed_dynamic(A, B, C_test, ts);
        end = std::chrono::high_resolution_clock::now();
        
        double t_dyn = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << std::left << std::setw(15) << ts 
                  << std::setw(15) << t_dyn 
                  << std::setw(15) << (t_naive / t_dyn) 
                  << (verify(C_naive, C_test) ? "[Valid]" : "[Invalid]") << std::endl;
    }

    // -------------------------------------------------------------------------
    // PART 3: MULTI-THREADING SCALING (SIMD + 2D GRID)
    // -------------------------------------------------------------------------
    std::cout << "\n--- Part 3: Multi-Threading Scaling (SIMD 2D Grid) ---\n";
    std::cout << std::left << std::setw(15) << "Threads" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(20) << "Speedup vs Naive"
              << std::setw(20) << "Scaling vs 1-Thr"
              << "Status" << std::endl;
    std::cout << "------------------------------------------------------------------------\n";

    // Determine hardware concurrency
    int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 8; // Fallback if undetectable

    // Define thread counts to test
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    // Add max_threads to the list if it's not already there
    if (std::find(thread_counts.begin(), thread_counts.end(), max_threads) == thread_counts.end()) {
        thread_counts.push_back(max_threads);
    }
    std::sort(thread_counts.begin(), thread_counts.end());

    SimdGemmStrategy simd_strategy(32); // Use the proven 32x32 optimal tile size
    double t_simd_multi_1 = 0; // Baseline for scaling efficiency

    for (int threads : thread_counts) {
        Matrix C_multi(M, N);
        
        // Run the parallel execution
        double t_multi = ParallelExecutor::run(simd_strategy, threads, A, B, C_multi);
        
        // Save the 1-thread time as a baseline for scaling efficiency
        if (threads == 1) t_simd_multi_1 = t_multi;

        // Calculate speedups
        double speedup_naive = t_naive / t_multi;
        double scaling_eff = (threads == 1) ? 1.0 : (t_simd_multi_1 / t_multi);

        std::cout << std::left << std::setw(15) << threads 
                  << std::setw(15) << t_multi 
                  << std::setw(20) << std::to_string(speedup_naive) + "x"
                  << std::setw(20) << std::to_string(scaling_eff) + "x"
                  << (verify(C_naive, C_multi) ? "[Valid]" : "[Invalid]") << std::endl;
    }

    std::cout << "=================================================================\n";
    return 0;
}