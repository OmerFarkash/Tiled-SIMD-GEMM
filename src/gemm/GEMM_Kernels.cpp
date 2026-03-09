#include "../../include/gemm/GEMM_Common.hpp"

// 1. Pure Naive: The "Baseline"
void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    int N = A.rows;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A.data[i * N + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
}

// 2. Naive Transpose: Transpose B first, then multiply
void gemm_transpose_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    int N = A.rows;
    Matrix BT(N, N);
    
    // Explicit full transpose
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            BT.data[j * N + i] = B.data[i * N + j];
        }
    }

    // Multiply using BT (Continuous access in both)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A.data[i * N + k] * BT.data[j * N + k];
            }
            C.data[i * N + j] = sum;
        }
    }
}

// 3. Tiled Packed: Prepares for Multi-processing
// Each "Job" is a tile of C. It packs B on the fly.
void gemm_tiled_packed(const Matrix& A, const Matrix& B, Matrix& C) {
    int N = A.rows;

    for (int i = 0; i < N; i += TILE_SIZE) {
        // Calculate remaining height to avoid overflow
        int valid_i = std::min(TILE_SIZE, N - i);

        for (int j = 0; j < N; j += TILE_SIZE) {
            int valid_j = std::min(TILE_SIZE, N - j);
            
            // Temporary SRAM-like buffer for the current tile of B
            float b_tile[TILE_SIZE * TILE_SIZE] = {0}; // Initialize with zeros

            for (int k = 0; k < N; k += TILE_SIZE) {
                int valid_k = std::min(TILE_SIZE, N - k);

                // --- PACKING STEP (With Tail Handling) ---
                // Load B into local transposed storage
                for (int rr = 0; rr < valid_k; ++rr) {
                    for (int cc = 0; cc < valid_j; ++cc) {
                        // Read from DDR, write to local SRAM-like buffer
                        b_tile[cc * TILE_SIZE + rr] = B.data[(k + rr) * N + (j + cc)];
                    }
                }

                // --- COMPUTE STEP (With Tail Handling) ---
                for (int ii = 0; ii < valid_i; ++ii) {
                    for (int jj = 0; jj < valid_j; ++jj) {
                        float partial_sum = 0;
                        for (int kk = 0; kk < valid_k; ++kk) {
                            // Multiply current row of A with current transposed row of B_tile
                            partial_sum += A.data[(i + ii) * N + (k + kk)] * b_tile[jj * TILE_SIZE + kk];
                        }
                        C.data[(i + ii) * N + (j + jj)] += partial_sum;
                    }
                }
            }
        }
    }
}