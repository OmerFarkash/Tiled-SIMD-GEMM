cmake_minimum_required(VERSION 3.10)
project(Tiled-SIMD-Matrix-Engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Include path for all headers
include_directories(include)

# --- Phase 1: GEMV App ---
# Keep it separate for now
set(GEMV_SOURCES
    apps/GEMV_Main.cpp
    src/gemv/WeightLoader.cpp
    src/gemv/Strategies.cpp
    src/core/ParallelExecutor.cpp
)

add_executable(gemv_app ${GEMV_SOURCES})
target_compile_options(gemv_app PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
target_link_libraries(gemv_app PRIVATE Threads::Threads)

# --- Phase 2: GEMM Benchmark ---
# Purely GEMM logic, no core dependency for now
set(GEMM_SOURCES
    apps/GEMM_Main.cpp
    src/gemm/GEMM_Kernels.cpp
)

add_executable(gemm_bench ${GEMM_SOURCES})
target_compile_options(gemm_bench PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
target_link_libraries(gemm_bench PRIVATE Threads::Threads)