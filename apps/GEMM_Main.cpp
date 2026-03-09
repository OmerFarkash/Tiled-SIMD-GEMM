#include "../include/gemm/GEMM_Common.hpp"
#include <iomanip>
#include <random>
#include <vector>
#include <string>

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

    std::cout << "\n======================================================\n";
    std::cout << "      Phase 3: SIMD & Broadcasting Benchmark          \n";
    std::cout << "      Profile: Deep Learning FC Layer (Unaligned)     \n";
    std::cout << "      Dimensions: M=" << M << ", K=" << K << ", N=" << N << "\n";
    std::cout << "======================================================\n\n";

    // Setup matrices
    Matrix A(M, K), B(K, N);
    Matrix C_naive(M, N), C_trans(M, N), C_packed(M, N), C_simd(M, N);

    // Deterministic initialization
    const uint32_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : A.data) val = dist(gen);
    for (auto& val : B.data) val = dist(gen);

    std::cout << "--- Part 1: Algorithmic Comparison ---\n";
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
    std::cout << std::left << std::setw(32) << "Tiled SIMD regular matrix (32)" 
              << std::setw(15) << t_simd 
              << (t_naive / t_simd) << "x" << std::endl;

    // Validation
    std::cout << "\nValidation: ";
    if (verify(C_naive, C_trans) && verify(C_naive, C_packed) && verify(C_naive, C_simd)) {
        std::cout << "[SUCCESS] All kernels produce identical results.\n\n";
    } else {
        std::cout << "[ERROR] Discrepancy found in results!\n\n";
        return 1;
    }

    std::cout << "--- Part 2: Cache Sweet-Spot Analysis (Tiled Packed) ---\n";
    std::cout << std::left << std::setw(15) << "Tile Size" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Speedup"
              << "Status" << std::endl;
    std::cout << "------------------------------------------------------\n";

    std::vector<int> tile_sizes = {16, 32, 64, 128, 256, 512};
    for (int ts : tile_sizes) {
        Matrix C_test(M, N); // Corrected dimensions
        start = std::chrono::high_resolution_clock::now();
        gemm_tiled_packed_dynamic(A, B, C_test, ts);
        end = std::chrono::high_resolution_clock::now();
        
        double t_dyn = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << std::left << std::setw(15) << ts 
                  << std::setw(15) << t_dyn 
                  << std::setw(15) << (t_naive / t_dyn) 
                  << (verify(C_naive, C_test) ? "[Valid]" : "[Invalid]") << std::endl;
    }
    
    std::cout << "======================================================\n";
    return 0;
}