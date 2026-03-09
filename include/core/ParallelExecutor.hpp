#pragma once
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>

// Includes for GEMV
#include "../gemv/ComputeStrategy.hpp"
#include "../gemv/WeightLoader.hpp"

// Includes for GEMM
#include "../gemm/GEMM_Strategy.hpp"
#include "../gemm/GEMM_Common.hpp"

class ParallelExecutor {
public:
    // GEMV Execution (1D row splitting)
    static double run(ComputeStrategy& strategy, int num_threads, int total_rows, 
                      float* matrix, float* inputs, float* output, int full_width);

    // GEMM Execution (2D grid splitting)
    static double run(GEMM_Strategy& strategy, int num_threads, 
                      const Matrix& A, const Matrix& B, Matrix& C);
};