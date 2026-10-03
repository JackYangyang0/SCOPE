#include <cuda_runtime.h>

#define TILE_SIZE 32
#define SHMEM_PAD 33

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][TILE_SIZE][SHMEM_PAD];
    __shared__ float Bs[2][TILE_SIZE][SHMEM_PAD];
    
    const int block_row = blockIdx.y * TILE_SIZE;
    const int block_col = blockIdx.x * TILE_SIZE;
    
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    
    const int global_row = block_row + thread_row;
    const int global_col = block_col + thread_col;
    
    float acc = 0.0f;
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    bool a_row_in_bounds = (global_row < M);
    bool b_col_in_bounds = (global_col < N);
    
    // Preload first tile into buffer 0
    const int k_start = 0;
    const int load_k_a = k_start + thread_col;
    const int load_k_b = k_start + thread_row;
    
    // Scalar loads for A - one element per thread
    if (a_row_in_bounds && load_k_a < K) {
        As[0][thread_row][thread_col] = A[global_row * K + load_k_a];
    } else {
        As[0][thread_row][thread_col] = 0.0f;
    }
    
    // Scalar loads for B - one element per thread
    if (b_col_in_bounds && load_k_b < K) {
        Bs[0][thread_row][thread_col] = B[load_k_b * N + global_col];
    } else {
        Bs[0][thread_row][thread_col] = 0.0f;
    }
    
    __syncthreads();
    
    int current_buffer = 0;
    
    // Main loop with double buffering - process all but last tile
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        const int next_tile = tile + 1;
        const int next_k_start = next_tile * TILE_SIZE;
        const int next_buffer = 1 - current_buffer;
        
        // Load next tile into the other buffer (asynchronous with computation)
        const int next_load_k_a = next_k_start + thread_col;
        const int next_load_k_b = next_k_start + thread_row;
        
        // Scalar loads for A
        if (a_row_in_bounds && next_load_k_a < K) {
            As[next_buffer][thread_row][thread_col] = A[global_row * K + next_load_k_a];
        } else {
            As[next_buffer][thread_row][thread_col] = 0.0f;
        }
        
        // Scalar loads for B
        if (b_col_in_bounds && next_load_k_b < K) {
            Bs[next_buffer][thread_row][thread_col] = B[next_load_k_b * N + global_col];
        } else {
            Bs[next_buffer][thread_row][thread_col] = 0.0f;
        }
        
        // Compute partial dot product with current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[current_buffer][thread_row][k] * Bs[current_buffer][k][thread_col];
        }
        
        __syncthreads();
        current_buffer = next_buffer;
    }
    
    // Final tile computation (no next load)
    if (num_tiles > 0) {
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[current_buffer][thread_row][k] * Bs[current_buffer][k][thread_col];
        }
    }
    
    // Write result with coalesced access
    if (a_row_in_bounds && b_col_in_bounds) {
        C[global_row * N + global_col] = acc;
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}