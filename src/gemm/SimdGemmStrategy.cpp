#include "../../include/gemm/SimdGemmStrategy.hpp"

// Forward declaration of the block kernel from GEMM_Kernels.cpp
void gemm_tiled_simd_block(const Matrix& A, const Matrix& B, Matrix& C, 
                           int start_row, int end_row, 
                           int start_col, int end_col, 
                           int tile_size);

SimdGemmStrategy::SimdGemmStrategy(int tile_size_) : tile_size(tile_size_) {}

void SimdGemmStrategy::execute(int start_row, int end_row, 
                               int start_col, int end_col, 
                               const Matrix& A, const Matrix& B, Matrix& C) {
    
    // Direct call to the mathematical kernel without any memory wrappers or copies!
    gemm_tiled_simd_block(A, B, C, start_row, end_row, start_col, end_col, tile_size);
}