#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Tile dimensions
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 8;
    
    // Thread block dimensions
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    
    // Each thread computes a 4x4 tile
    constexpr int THREAD_M = 4;
    constexpr int THREAD_N = 4;
    
    // Shared memory for tiles (double buffered)
    extern __shared__ float shared_mem[];
    float* As[2] = {shared_mem, shared_mem + TILE_M * TILE_K};
    float* Bs[2] = {shared_mem + 2 * TILE_M * TILE_K, shared_mem + 2 * TILE_M * TILE_K + TILE_K * TILE_N};
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Global base row and column indices for the thread block
    int block_row = by * TILE_M;
    int block_col = bx * TILE_N;
    
    // Accumulators for the 4x4 output tile
    float acc[THREAD_M][THREAD_N] = {{0.0f}};
    
    // Double buffering state
    int buffer_idx = 0;
    
    // Load first tiles into shared memory
    int k_start = 0;
    int k_end = min(k_start + TILE_K, K);
    
    // Load A tile (TILE_M x TILE_K)
    for (int i = 0; i < THREAD_M; i++) {
        int global_row = block_row + ty * THREAD_M + i;
        if (global_row < M && k_start < K) {
            for (int k = 0; k < TILE_K; k += 4) {
                int global_k = k_start + k;
                if (global_k < K) {
                    if (global_k + 3 < K) {
                        float4 a_val = *((float4*)&A[global_row * K + global_k]);
                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k] = a_val.x;
                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + 1] = a_val.y;
                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + 2] = a_val.z;
                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + 3] = a_val.w;
                    } else {
                        for (int kk = 0; kk < 4; kk++) {
                            if (global_k + kk < K) {
                                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + kk] = A[global_row * K + global_k + kk];
                            } else {
                                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + kk] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int kk = 0; kk < 4; kk++) {
                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + kk] = 0.0f;
                    }
                }
            }
        } else {
            for (int k = 0; k < TILE_K; k++) {
                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k] = 0.0f;
            }
        }
    }
    
    // Load B tile (TILE_K x TILE_N)
    for (int j = 0; j < THREAD_N; j++) {
        int global_col = block_col + tx * THREAD_N + j;
        if (global_col < N && k_start < K) {
            for (int k = 0; k < TILE_K; k++) {
                int global_k = k_start + k;
                if (global_k < K) {
                    Bs[buffer_idx][k * TILE_N + (tx * THREAD_N + j)] = B[global_k * N + global_col];
                } else {
                    Bs[buffer_idx][k * TILE_N + (tx * THREAD_N + j)] = 0.0f;
                }
            }
        } else {
            for (int k = 0; k < TILE_K; k++) {
                Bs[buffer_idx][k * TILE_N + (tx * THREAD_N + j)] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // Main GEMM loop
    for (int k = 0; k < K; k += TILE_K) {
        int next_k = k + TILE_K;
        bool last_iter = (next_k >= K);
        
        // Compute partial products for current tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    acc[i][j] += As[buffer_idx][(ty * THREAD_M + i) * TILE_K + kk] * 
                                 Bs[buffer_idx][kk * TILE_N + (tx * THREAD_N + j)];
                }
            }
        }
        
        if (!last_iter) {
            // Switch buffer
            buffer_idx = 1 - buffer_idx;
            
            // Load next A tile
            k_start = next_k;
            k_end = min(k_start + TILE_K, K);
            
            for (int i = 0; i < THREAD_M; i++) {
                int global_row = block_row + ty * THREAD_M + i;
                if (global_row < M && k_start < K) {
                    for (int k = 0; k < TILE_K; k += 4) {
                        int global_k = k_start + k;
                        if (global_k < K) {
                            if (global_k + 3 < K) {
                                float4 a_val = *((float4*)&A[global_row * K + global_k]);
                                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k] = a_val.x;
                                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + 1] = a_val.y;
                                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + 2] = a_val.z;
                                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + 3] = a_val.w;
                            } else {
                                for (int kk = 0; kk < 4; kk++) {
                                    if (global_k + kk < K) {
                                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + kk] = A[global_row * K + global_k + kk];
                                    } else {
                                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + kk] = 0.0f;
                                    }
                                }
                            }
                        } else {
                            for (int kk = 0; kk < 4; kk++) {
                                As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k + kk] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int k = 0; k < TILE_K; k++) {
                        As[buffer_idx][(ty * THREAD_M + i) * TILE_K + k] = 0.0f;
                    }
                }
            }
            
            // Load next B tile
            for (int j = 0; j < THREAD_N; j++) {
                int global_col = block_col + tx * THREAD_N + j;
                if (global_col < N && k_start < K) {
                    for (int k = 0; k < TILE_K; k++) {
                        int global_k = k_start + k;
                        if (global_k < K) {
                            Bs[buffer_idx][k * TILE_N + (tx * THREAD_N + j)] = B[global_k * N + global_col];
                        } else {
                            Bs[buffer_idx][k * TILE_N + (tx * THREAD_N + j)] = 0.0f;
                        }
                    }
                } else {
                    for (int k = 0; k < TILE_K; k++) {
                        Bs[buffer_idx][k * TILE_N + (tx * THREAD_N + j)] = 0.0f;
                    }
                }
            }
            
            __syncthreads();
        }
    }
    
    // Write results back to global memory
    for (int i = 0; i < THREAD_M; i++) {
        for (int j = 0; j < THREAD_N; j++) {
            int out_row = block_row + ty * THREAD_M + i;
            int out_col = block_col + tx * THREAD_N + j;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[i][j];
            }
        }
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    // Launch configuration
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    
    dim3 blockDim(BLOCK_N, BLOCK_M);
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    // Shared memory size calculation
    constexpr int TILE_K = 8;
    size_t shared_mem_size = 2 * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    
    // Launch kernel
    gemm_kernel<<<gridDim, blockDim, shared_mem_size>>>(A, B, C, M, N, K);
}
