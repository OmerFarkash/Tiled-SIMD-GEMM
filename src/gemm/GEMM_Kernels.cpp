#include "../../include/gemm/GEMM_Common.hpp"

// 1. Pure Naive (M x K) * (K x N)
void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
}

// 2. Naive Transpose
void gemm_transpose_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    // BT will be N x K
    Matrix BT(N, K);
    
    // Transpose B (K x N) into BT (N x K)
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            BT.data[j * K + k] = B.data[k * N + j];
        }
    }

    // Multiply A (M x K) with BT (N x K)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * BT.data[j * K + k];
            }
            C.data[i * N + j] = sum;
        }
    }
}
/* * PHASE 2: Tiled Packed (Scalar / Dot-Product)
 * Logic: Transpose B tile to SRAM for linear access.
 */
void gemm_tiled_packed_dynamic(const Matrix& A, const Matrix& B, Matrix& C, int tile_size) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    // Allocate buffer once
    std::vector<float> b_tile(tile_size * tile_size, 0.0f);

    for (int i = 0; i < M; i += tile_size) {
        int valid_i = std::min(tile_size, M - i);
        
        for (int j = 0; j < N; j += tile_size) {
            int valid_j = std::min(tile_size, N - j);

            for (int k = 0; k < K; k += tile_size) {
                int valid_k = std::min(tile_size, K - k);

                // --- PACKING STEP ---
                // Load chunk of B (valid_k x valid_j) into transposed b_tile
                for (int rr = 0; rr < valid_k; ++rr) {
                    for (int cc = 0; cc < valid_j; ++cc) {
                        b_tile[cc * tile_size + rr] = B.data[(k + rr) * N + (j + cc)];
                    }
                }

                // --- COMPUTE STEP ---
                for (int ii = 0; ii < valid_i; ++ii) {
                    for (int jj = 0; jj < valid_j; ++jj) {
                        float partial_sum = 0;
                        for (int kk = 0; kk < valid_k; ++kk) {
                            partial_sum += A.data[(i + ii) * K + (k + kk)] * b_tile[jj * tile_size + kk];
                        }
                        C.data[(i + ii) * N + (j + jj)] += partial_sum;
                    }
                }
            }
        }
    }
}

// Keep the fixed TILE_SIZE version updated too, exactly like the dynamic one 
// but replacing tile_size with TILE_SIZE macro.
void gemm_tiled_packed(const Matrix& A, const Matrix& B, Matrix& C) {
    gemm_tiled_packed_dynamic(A, B, C, TILE_SIZE);
}

/* * PHASE 3: Tiled SIMD (Vectorized / Outer-Product)
 * Logic: Use _mm256_broadcast_ps. NO transpose needed for B tile.
 */
// SIMD Kernel with Broadcasting and Robust Tail Handling
void gemm_tiled_simd(const Matrix& A, const Matrix& B, Matrix& C, int tile_size) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    // Buffer for the B tile (Notice: NO transpose needed for broadcasting!)
    std::vector<float> b_tile(tile_size * tile_size, 0.0f);

    for (int i = 0; i < M; i += tile_size) {
        int valid_i = std::min(tile_size, M - i);
        
        for (int j = 0; j < N; j += tile_size) {
            int valid_j = std::min(tile_size, N - j);

            for (int k = 0; k < K; k += tile_size) {
                int valid_k = std::min(tile_size, K - k);

                // --- PACKING STEP (Row-Major, no transpose) ---
                for (int rr = 0; rr < valid_k; ++rr) {
                    for (int cc = 0; cc < valid_j; ++cc) {
                        b_tile[rr * tile_size + cc] = B.data[(k + rr) * N + (j + cc)];
                    }
                }

                // --- SIMD COMPUTE STEP ---
                for (int ii = 0; ii < valid_i; ++ii) {
                    // Process 8 columns of C simultaneously
                    for (int jj = 0; jj < valid_j; jj += 8) {
                        
                        // Check if we have a full 8-element block for SIMD
                        if (jj + 8 <= valid_j) {
                            // Load 8 elements from C
                            __m256 c_vec = _mm256_loadu_ps(&C.data[(i + ii) * N + (j + jj)]);

                            for (int kk = 0; kk < valid_k; ++kk) {
                                // Broadcast 1 element from A to all 8 slots
                                __m256 a_val = _mm256_set1_ps(A.data[(i + ii) * K + (k + kk)]);
                                
                                // Load 8 elements from the packed B tile
                                __m256 b_vec = _mm256_loadu_ps(&b_tile[kk * tile_size + jj]);
                                
                                // FMA: c_vec = a_val * b_vec + c_vec
                                c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                            }
                            
                            // Store the updated 8 elements back to C
                            _mm256_storeu_ps(&C.data[(i + ii) * N + (j + jj)], c_vec);
                        } 
                        else {
                            // --- SCALAR TAIL HANDLING (Fallback for remaining < 8 elements) ---
                            for (int rem_j = jj; rem_j < valid_j; ++rem_j) {
                                float partial_sum = 0;
                                for (int kk = 0; kk < valid_k; ++kk) {
                                    partial_sum += A.data[(i + ii) * K + (k + kk)] * b_tile[kk * tile_size + rem_j];
                                }
                                C.data[(i + ii) * N + (j + rem_j)] += partial_sum;
                            }
                        }
                    }
                }
            }
        }
    }
}