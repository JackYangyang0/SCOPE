#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MICRO_TILE_M 2
#define MICRO_TILE_N 2
#define THREADS_PER_BLOCK 256

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Shared memory with padding to avoid bank conflicts (32 banks -> +1 padding)
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = &shared_mem[TILE_SIZE * (TILE_SIZE + 1)];
    
    // Thread mapping: 256 threads = 16x16 logical layout.
    int tid = threadIdx.x;
    int tx = tid % 16;
    int ty = tid / 16;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int cRowBase = by * TILE_SIZE;
    int cColBase = bx * TILE_SIZE;
    
    // Each thread computes an 8x8 micro-tile
    float cAccum[MICRO_TILE_M][MICRO_TILE_N] = {{0.0f}};
    
    int numKTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Main loop over K tiles
    for (int kTile = 0; kTile < numKTiles; ++kTile) {
        for (int idx = tid; idx < TILE_SIZE * TILE_SIZE; idx += THREADS_PER_BLOCK) {
            int row = idx / TILE_SIZE;
            int col = idx % TILE_SIZE;
            int globalRow = cRowBase + row;
            int globalK = kTile * TILE_SIZE + col;
            As[row * (TILE_SIZE + 1) + col] =
                (globalRow < M && globalK < K) ? A[globalRow * K + globalK] : 0.0f;

            int globalCol = cColBase + col;
            int bGlobalK = kTile * TILE_SIZE + row;
            Bs[col * (TILE_SIZE + 1) + row] =
                (bGlobalK < K && globalCol < N) ? B[bGlobalK * N + globalCol] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_M; ++i) {
                float aVal = As[(ty * MICRO_TILE_M + i) * (TILE_SIZE + 1) + k];
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_N; ++j) {
                    float bVal = Bs[(tx * MICRO_TILE_N + j) * (TILE_SIZE + 1) + k];
                    cAccum[i][j] += aVal * bVal;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < MICRO_TILE_M; ++i) {
        int cRow = cRowBase + ty * MICRO_TILE_M + i;
        if (cRow < M) {
            #pragma unroll
            for (int j = 0; j < MICRO_TILE_N; ++j) {
            int cCol = cColBase + tx * MICRO_TILE_N + j;
                if (cCol < N) {
                    C[cRow * N + cCol] = cAccum[i][j];
                }
            }
        }
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Shared memory size: 2 tiles with padding
    size_t shared_mem_size = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);
    
    // Launch kernel with dynamic shared memory
    gemm_kernel<<<gridDim, blockDim, shared_mem_size>>>(A, B, C, M, N, K);
}
