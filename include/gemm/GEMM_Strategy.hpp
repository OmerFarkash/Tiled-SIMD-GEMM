#pragma once
#include "GEMM_Common.hpp"

// Abstract base class for GEMM execution strategies
class GEMM_Strategy {
public:
    virtual ~GEMM_Strategy() = default;

    // Executes a specific 2D block of the C matrix directly using Matrix references
    virtual void execute(int start_row, int end_row, 
                         int start_col, int end_col, 
                         const Matrix& A, const Matrix& B, Matrix& C) = 0;
};