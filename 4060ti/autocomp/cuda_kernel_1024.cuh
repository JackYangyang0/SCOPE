#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block size: 16x16 threads
    // Each thread computes 2x2 elements
    const int BLOCK_SIZE = 16;
    const int TILE_SIZE = 32; // Shared memory tile size

    // Shared memory for double buffering
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate starting positions in global memory
    int cRow0 = by * TILE_SIZE + ty * 2;
    int cRow1 = cRow0 + 1;
    int cCol0 = bx * TILE_SIZE + tx * 2;
    int cCol1 = cCol0 + 1;

    // Initialize accumulators for 2x2 sub-tile
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    // Main computation loop
    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load A tile into shared memory
        int aRow0 = by * TILE_SIZE + ty * 2;
        int aRow1 = aRow0 + 1;
        int aCol = k + tx * 2;
        
        // Load 2x2 elements for A
        if (aRow0 < M && aCol < K) {
            As[ty * 2][tx * 2] = A[aRow0 * K + aCol];
        } else {
            As[ty * 2][tx * 2] = 0.0f;
        }
        if (aRow0 < M && aCol + 1 < K) {
            As[ty * 2][tx * 2 + 1] = A[aRow0 * K + aCol + 1];
        } else {
            As[ty * 2][tx * 2 + 1] = 0.0f;
        }
        if (aRow1 < M && aCol < K) {
            As[ty * 2 + 1][tx * 2] = A[aRow1 * K + aCol];
        } else {
            As[ty * 2 + 1][tx * 2] = 0.0f;
        }
        if (aRow1 < M && aCol + 1 < K) {
            As[ty * 2 + 1][tx * 2 + 1] = A[aRow1 * K + aCol + 1];
        } else {
            As[ty * 2 + 1][tx * 2 + 1] = 0.0f;
        }
        
        // Load B tile into shared memory
        int bRow = k + ty * 2;
        int bCol0 = bx * TILE_SIZE + tx * 2;
        int bCol1 = bCol0 + 1;
        
        // Load 2x2 elements for B
        if (bRow < K && bCol0 < N) {
            Bs[ty * 2][tx * 2] = B[bRow * N + bCol0];
        } else {
            Bs[ty * 2][tx * 2] = 0.0f;
        }
        if (bRow < K && bCol1 < N) {
            Bs[ty * 2][tx * 2 + 1] = B[bRow * N + bCol1];
        } else {
            Bs[ty * 2][tx * 2 + 1] = 0.0f;
        }
        if (bRow + 1 < K && bCol0 < N) {
            Bs[ty * 2 + 1][tx * 2] = B[(bRow + 1) * N + bCol0];
        } else {
            Bs[ty * 2 + 1][tx * 2] = 0.0f;
        }
        if (bRow + 1 < K && bCol1 < N) {
            Bs[ty * 2 + 1][tx * 2 + 1] = B[(bRow + 1) * N + bCol1];
        } else {
            Bs[ty * 2 + 1][tx * 2 + 1] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc00 += As[ty * 2][i] * Bs[i][tx * 2];
            acc01 += As[ty * 2][i] * Bs[i][tx * 2 + 1];
            acc10 += As[ty * 2 + 1][i] * Bs[i][tx * 2];
            acc11 += As[ty * 2 + 1][i] * Bs[i][tx * 2 + 1];
        }
        
        __syncthreads();
    }

    // Write results back to global memory
    if (cRow0 < M && cCol0 < N) C[cRow0 * N + cCol0] = acc00;
    if (cRow0 < M && cCol1 < N) C[cRow0 * N + cCol1] = acc01;
    if (cRow1 < M && cCol0 < N) C[cRow1 * N + cCol0] = acc10;
    if (cRow1 < M && cCol1 < N) C[cRow1 * N + cCol1] = acc11;
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int BLOCK_SIZE = 16;
    const int TILE_SIZE = 32;
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}