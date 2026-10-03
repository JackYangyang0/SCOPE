#include <cuda_runtime.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define TILE_K_PADDED (TILE_K + 1)

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K_PADDED];
    __shared__ float Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int cRow = blockRow * TILE_M + ty * 4;
    int cCol = blockCol * TILE_N + tx * 4;

    float acc[4][4] = {{0.0f}};

    // Since M=N=K=1024 is multiple of TILE sizes, no cleanup needed
    for (int tile = 0; tile < K; tile += TILE_K) {
        // Collaborative loading of A: all threads participate
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int globalRow = cRow + i;
            int k_start = tx * 2;
            // Load 2 elements per thread for A (since TILE_K=32, 32/16=2)
            if (k_start < TILE_K) {
                As[ty * 4 + i][k_start] = A[globalRow * K + tile + k_start];
            }
            if (k_start + 1 < TILE_K) {
                As[ty * 4 + i][k_start + 1] = A[globalRow * K + tile + k_start + 1];
            }
        }

        // Collaborative loading of B: all threads participate
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int globalCol = cCol + j;
            int k_start = ty * 2;
            // Load 2 elements per thread for B
            if (k_start < TILE_K) {
                Bs[k_start][tx * 4 + j] = B[(tile + k_start) * N + globalCol];
            }
            if (k_start + 1 < TILE_K) {
                Bs[k_start + 1][tx * 4 + j] = B[(tile + k_start + 1) * N + globalCol];
            }
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll 4
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += As[ty * 4 + i][k] * Bs[k][tx * 4 + j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory (no bounds checking needed for 1024x1024)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            C[(cRow + i) * N + (cCol + j)] = acc[i][j];
        }
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}