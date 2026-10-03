#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 8;
    
    __shared__ float As[BLOCK_M][32];
    __shared__ float Bs[32][BLOCK_N];
    
    const int blockRow = blockIdx.y * BLOCK_M;
    const int blockCol = blockIdx.x * BLOCK_N;
    
    const int threadRow = threadIdx.y; // 0-31 for BLOCK_M=32
    const int threadCol = threadIdx.x; // 0-7 for BLOCK_N=8
    
    float acc = 0.0f;
    
    for (int tileIdx = 0; tileIdx < (K + 31) / 32; ++tileIdx) {
        // Load A tile: each thread loads 4 elements (32 cols / 8 threads = 4)
        int aRow = blockRow + threadRow;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int aCol = tileIdx * 32 + threadCol + i * 8;
            if (aRow < M && aCol < K) {
                As[threadRow][threadCol + i * 8] = A[aRow * K + aCol];
            } else {
                As[threadRow][threadCol + i * 8] = 0.0f;
            }
        }
        
        // Load B tile: each thread loads 1 element
        int bRow = tileIdx * 32 + threadRow;
        int bCol = blockCol + threadCol;
        Bs[threadRow][threadCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        
        __syncthreads();
        
        // Compute partial products
        for (int k = 0; k < 32; ++k) {
            acc += As[threadRow][k] * Bs[k][threadCol];
        }
        
        __syncthreads();
    }
    
    // Write result
    int row = blockRow + threadRow;
    int col = blockCol + threadCol;
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int BLOCK_M = 32;
    const int BLOCK_N = 8;
    
    dim3 blockDim(BLOCK_N, BLOCK_M); // x=BLOCK_N=8, y=BLOCK_M=32
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}