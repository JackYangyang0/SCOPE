#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block size: 16x16 threads
    const int BLOCK_SIZE = 16;
    const int TILE_SIZE = 32; // Shared memory tile dimension
    
    // Each thread computes a 2x2 sub-tile of the output
    const int THREAD_ROWS = 2;
    const int THREAD_COLS = 2;
    
    // Shared memory tiles for A and B with double buffering
    extern __shared__ float shared_mem[];
    float* sA[2] = {
        shared_mem,
        shared_mem + TILE_SIZE * TILE_SIZE
    };
    float* sB[2] = {
        shared_mem + 2 * TILE_SIZE * TILE_SIZE,
        shared_mem + 3 * TILE_SIZE * TILE_SIZE
    };
    
    // Thread and block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global memory indices for this thread's output elements
    int row = by * BLOCK_SIZE * THREAD_ROWS + ty * THREAD_ROWS;
    int col = bx * BLOCK_SIZE * THREAD_COLS + tx * THREAD_COLS;
    
    // Accumulators for the 2x2 output sub-tile
    float acc[THREAD_ROWS][THREAD_COLS] = {{0.0f}};
    
    // Double buffering state
    int current_buffer = 0;
    int next_buffer = 1;
    
    // Load first tiles into shared memory
    int global_k = 0;
    
    // Load initial tiles
    if (global_k < K) {
        // Load A tile: TILE_SIZE x TILE_SIZE
        int a_row = by * TILE_SIZE + ty;
        int a_col = global_k + tx;
        if (a_row < M && a_col < K) {
            sA[current_buffer][ty * TILE_SIZE + tx] = A[a_row * K + a_col];
        } else {
            sA[current_buffer][ty * TILE_SIZE + tx] = 0.0f;
        }
        
        // Load B tile: TILE_SIZE x TILE_SIZE
        int b_row = global_k + ty;
        int b_col = bx * TILE_SIZE + tx;
        if (b_row < K && b_col < N) {
            sB[current_buffer][ty * TILE_SIZE + tx] = B[b_row * N + b_col];
        } else {
            sB[current_buffer][ty * TILE_SIZE + tx] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Main computation loop
    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load next tiles while computing current
        if (k + TILE_SIZE < K) {
            int next_k = k + TILE_SIZE;
            
            // Load next A tile
            int a_row = by * TILE_SIZE + ty;
            int a_col = next_k + tx;
            if (a_row < M && a_col < K) {
                sA[next_buffer][ty * TILE_SIZE + tx] = A[a_row * K + a_col];
            } else {
                sA[next_buffer][ty * TILE_SIZE + tx] = 0.0f;
            }
            
            // Load next B tile
            int b_row = next_k + ty;
            int b_col = bx * TILE_SIZE + tx;
            if (b_row < K && b_col < N) {
                sB[next_buffer][ty * TILE_SIZE + tx] = B[b_row * N + b_col];
            } else {
                sB[next_buffer][ty * TILE_SIZE + tx] = 0.0f;
            }
        }
        
        // Compute partial products for the current tile
        for (int kk = 0; kk < TILE_SIZE; ++kk) {
            // Broadcast values from shared memory
            float a_val[THREAD_ROWS];
            float b_val[THREAD_COLS];
            
            // Load A values for this thread's rows
            a_val[0] = sA[current_buffer][(ty * THREAD_ROWS + 0) * TILE_SIZE + kk];
            a_val[1] = sA[current_buffer][(ty * THREAD_ROWS + 1) * TILE_SIZE + kk];
            
            // Load B values for this thread's columns
            b_val[0] = sB[current_buffer][kk * TILE_SIZE + (tx * THREAD_COLS + 0)];
            b_val[1] = sB[current_buffer][kk * TILE_SIZE + (tx * THREAD_COLS + 1)];
            
            // Accumulate products
            acc[0][0] += a_val[0] * b_val[0];
            acc[0][1] += a_val[0] * b_val[1];
            acc[1][0] += a_val[1] * b_val[0];
            acc[1][1] += a_val[1] * b_val[1];
        }
        
        __syncthreads();
        
        // Swap buffers
        int temp = current_buffer;
        current_buffer = next_buffer;
        next_buffer = temp;
    }
    
    // Write results to global memory
    if (row + 0 < M && col + 0 < N) C[(row + 0) * N + (col + 0)] = acc[0][0];
    if (row + 0 < M && col + 1 < N) C[(row + 0) * N + (col + 1)] = acc[0][1];
    if (row + 1 < M && col + 0 < N) C[(row + 1) * N + (col + 0)] = acc[1][0];
    if (row + 1 < M && col + 1 < N) C[(row + 1) * N + (col + 1)] = acc[1][1];
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int BLOCK_SIZE = 16;
    const int TILE_SIZE = 32;
    
    // Calculate grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2), 
                 (M + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    
    // Shared memory size: 4 tiles of TILE_SIZE x TILE_SIZE floats
    size_t shared_mem_size = 4 * TILE_SIZE * TILE_SIZE * sizeof(float);
    
    // Launch kernel
    gemm_kernel<<<gridDim, blockDim, shared_mem_size>>>(A, B, C, M, N, K);
}