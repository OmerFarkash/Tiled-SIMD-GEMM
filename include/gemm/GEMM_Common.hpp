#ifndef GEMM_COMMON_HPP
#define GEMM_COMMON_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

// Define tile size for the packed version
// 64x64 is usually a sweet spot for L1/L2 cache
const int TILE_SIZE = 64;

struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
};

#endif