#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices for the output tile
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Accumulator for this thread's contribution
    float c_reg = 0.0f;
    
    // Loop over tiles in K dimension
    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
        // Load A tile: each thread loads one element
        if (row < M && (k0 + tx) < K) {
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile: each thread loads one element  
        if ((k0 + ty) < K && col < N) {
            Bs[ty][tx] = B[(k0 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            c_reg += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = c_reg;
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}