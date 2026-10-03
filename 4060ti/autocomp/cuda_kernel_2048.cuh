#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block size: 16x16 threads
    const int BLOCK_M = 16;
    const int BLOCK_N = 16;
    const int TILE_M = 32;
    const int TILE_N = 32;
    
    // Shared memory tiles
    __shared__ float As[TILE_M][TILE_N];
    __shared__ float Bs[TILE_N][TILE_M];
    
    // Thread indices
    int tx = threadIdx.x; // 0-15
    int ty = threadIdx.y; // 0-15
    
    // Global block indices
    int bx = blockIdx.x; // for M dimension
    int by = blockIdx.y; // for N dimension
    
    // Register accumulation: 2x2 per thread
    float acc[2][2] = {0.0f};
    
    // Loop over tiles in K dimension
    for (int tile_k = 0; tile_k < K; tile_k += TILE_N) {
        // Load A tile: TILE_M x TILE_N
        // Each thread loads 2 elements (32*32 / (16*16) = 4, but we do 2 loads per thread)
        int a_row0 = ty * 2 + 0;
        int a_col0 = tx * 2 + 0;
        int a_row1 = ty * 2 + 1;
        int a_col1 = tx * 2 + 1;
        
        int global_a_row0 = bx * TILE_M + a_row0;
        int global_a_col0 = tile_k + a_col0;
        int global_a_row1 = bx * TILE_M + a_row1;
        int global_a_col1 = tile_k + a_col1;
        
        if (global_a_row0 < M && global_a_col0 < K) {
            As[a_row0][tx * 2 + 0] = A[global_a_row0 * K + global_a_col0];
        } else {
            As[a_row0][tx * 2 + 0] = 0.0f;
        }
        
        if (global_a_row0 < M && global_a_col1 < K) {
            As[a_row0][tx * 2 + 1] = A[global_a_row0 * K + global_a_col1];
        } else {
            As[a_row0][tx * 2 + 1] = 0.0f;
        }
        
        if (global_a_row1 < M && global_a_col0 < K) {
            As[a_row1][tx * 2 + 0] = A[global_a_row1 * K + global_a_col0];
        } else {
            As[a_row1][tx * 2 + 0] = 0.0f;
        }
        
        if (global_a_row1 < M && global_a_col1 < K) {
            As[a_row1][tx * 2 + 1] = A[global_a_row1 * K + global_a_col1];
        } else {
            As[a_row1][tx * 2 + 1] = 0.0f;
        }
        
        // Load B tile: TILE_N x TILE_M
        int b_row0 = tx * 2 + 0;
        int b_col0 = ty * 2 + 0;
        int b_row1 = tx * 2 + 1;
        int b_col1 = ty * 2 + 1;
        
        int global_b_row0 = tile_k + b_row0;
        int global_b_col0 = by * TILE_N + b_col0;
        int global_b_row1 = tile_k + b_row1;
        int global_b_col1 = by * TILE_N + b_col1;
        
        if (global_b_row0 < K && global_b_col0 < N) {
            Bs[b_row0][ty * 2 + 0] = B[global_b_row0 * N + global_b_col0];
        } else {
            Bs[b_row0][ty * 2 + 0] = 0.0f;
        }
        
        if (global_b_row1 < K && global_b_col0 < N) {
            Bs[b_row1][ty * 2 + 0] = B[global_b_row1 * N + global_b_col0];
        } else {
            Bs[b_row1][ty * 2 + 0] = 0.0f;
        }
        
        if (global_b_row0 < K && global_b_col1 < N) {
            Bs[b_row0][ty * 2 + 1] = B[global_b_row0 * N + global_b_col1];
        } else {
            Bs[b_row0][ty * 2 + 1] = 0.0f;
        }
        
        if (global_b_row1 < K && global_b_col1 < N) {
            Bs[b_row1][ty * 2 + 1] = B[global_b_row1 * N + global_b_col1];
        } else {
            Bs[b_row1][ty * 2 + 1] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot products
        for (int k = 0; k < TILE_N; k++) {
            // First row of A fragment
            float a_val0 = As[ty * 2 + 0][k];
            float a_val1 = As[ty * 2 + 1][k];
            
            // First column of B fragment
            float b_val0 = Bs[k][tx * 2 + 0];
            float b_val1 = Bs[k][tx * 2 + 1];
            
            acc[0][0] += a_val0 * b_val0;
            acc[0][1] += a_val0 * b_val1;
            acc[1][0] += a_val1 * b_val0;
            acc[1][1] += a_val1 * b_val1;
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    int c_row0 = bx * TILE_M + ty * 2 + 0;
    int c_row1 = bx * TILE_M + ty * 2 + 1;
    int c_col0 = by * TILE_N + tx * 2 + 0;
    int c_col1 = by * TILE_N + tx * 2 + 1;
    
    if (c_row0 < M && c_col0 < N) {
        C[c_row0 * N + c_col0] = acc[0][0];
    }
    if (c_row0 < M && c_col1 < N) {
        C[c_row0 * N + c_col1] = acc[0][1];
    }
    if (c_row1 < M && c_col0 < N) {
        C[c_row1 * N + c_col0] = acc[1][0];
    }
    if (c_row1 < M && c_col1 < N) {
        C[c_row1 * N + c_col1] = acc[1][1];
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int BLOCK_M = 16;
    const int BLOCK_N = 16;
    
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_M, BLOCK_N);
    
    gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}