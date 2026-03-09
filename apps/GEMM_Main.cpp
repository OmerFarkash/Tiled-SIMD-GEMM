#include "../include/gemm/GEMM_Common.hpp"
#include <iomanip>
#include <random>

// Function declarations
void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_transpose_naive(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed(const Matrix& A, const Matrix& B, Matrix& C);

bool verify(const Matrix& C1, const Matrix& C2) {
    for (size_t i = 0; i < C1.data.size(); ++i) {
        if (std::abs(C1.data[i] - C2.data[i]) > 1e-3) return false;
    }
    return true;
}

int main() {
    const int N = 1025; 
    std::cout << "--- GEMM Cache-Locality Benchmark (" << N << "x" << N << ") ---" << std::endl;

    Matrix A(N, N), B(N, N), C_naive(N, N), C_trans(N, N), C_packed(N, N);

    // Initialize with random data by seed
    const uint32_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        A.data[i] = dist(gen);
        B.data[i] = dist(gen);
    }

    // 1. Benchmark Naive
    auto start = std::chrono::high_resolution_clock::now();
    gemm_naive(A, B, C_naive);
    auto end = std::chrono::high_resolution_clock::now();
    double t_naive = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::left << std::setw(25) << "Naive GEMM:" 
              << std::fixed << std::setprecision(2) << t_naive << " ms (Baseline)" << std::endl;

    // 2. Benchmark Naive Transpose
    start = std::chrono::high_resolution_clock::now();
    gemm_transpose_naive(A, B, C_trans);
    end = std::chrono::high_resolution_clock::now();
    double t_trans = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::left << std::setw(25) << "Naive Transpose:" 
              << t_trans << " ms (Speedup: " << t_naive / t_trans << "x)" << std::endl;

    // 3. Benchmark Tiled Packed
    start = std::chrono::high_resolution_clock::now();
    gemm_tiled_packed(A, B, C_packed);
    end = std::chrono::high_resolution_clock::now();
    double t_packed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::left << std::setw(25) << "Tiled Packed:" 
              << t_packed << " ms (Speedup: " << t_naive / t_packed << "x)" << std::endl;

    std::cout << "----------------------------------------------" << std::endl;

    // Verification
    if (verify(C_naive, C_trans) && verify(C_naive, C_packed)) {
        std::cout << "SUCCESS: All results verified!" << std::endl;
    } else {
        std::cout << "ERROR: Discrepancy in results!" << std::endl;
    }

    return 0;
}