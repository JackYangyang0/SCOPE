#include <cuda_runtime.h>

// Optimized for H800 (sm_90) while maintaining RTX 2060 (sm_75) compatibility
// Balanced tile sizes that fit within 64KB shared memory limit on sm_75
#define TILE_M 128
#define TILE_N 128
#define TILE_K 32

// Block dimensions: 32x16 = 512 threads (optimal for H800, compatible with sm_75)
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 16

// Thread-level work: each thread computes 8x4 = 32 outputs (reduced register pressure)
#define THREAD_TILE_M (TILE_M / THREADS_PER_BLOCK_Y) // 8
#define THREAD_TILE_N (TILE_N / THREADS_PER_BLOCK_X) // 4

// Padding for bank conflict avoidance (H800 has 128 banks, sm_75 has 32 banks)
// For B: TILE_N=128, with 128 banks -> no padding needed if we align properly
// But to be safe across architectures and avoid any stride conflicts, pad to next power of 2
#define PAD_A 0
#define PAD_B 0
#define TILE_N_PADDED (TILE_N + PAD_B)

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    extern __shared__ float shared_mem[];
    
    // Double-buffered shared memory layout: [2][A_tile] and [2][B_tile]
    float (*As)[TILE_M][TILE_K + PAD_A] =
        reinterpret_cast<float (*)[TILE_M][TILE_K + PAD_A]>(shared_mem);
    float (*Bs)[TILE_K][TILE_N_PADDED] =
        reinterpret_cast<float (*)[TILE_K][TILE_N_PADDED]>(
            &shared_mem[2 * TILE_M * (TILE_K + PAD_A)]);
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0..31
    int ty = threadIdx.y; // 0..15
    
    // Each thread accumulates THREAD_TILE_M x THREAD_TILE_N values
    float c_local[THREAD_TILE_M][THREAD_TILE_N] = {{0.0f}};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    // Double buffering indices
    int curr_buffer = 0;
    
    // Load first tile into buffer 0
    int a_row_base = by * TILE_M;
    int b_col_base = bx * TILE_N;
    
    // Prefetch first tile - coalesced loads using vectorization
    #pragma unroll
    for (int mi = 0; mi < THREAD_TILE_M; ++mi) {
        int global_row = a_row_base + ty + mi * THREADS_PER_BLOCK_Y;
        #pragma unroll
        for (int k = 0; k < TILE_K; k += 4) {
            float4 a4;
            if (global_row < M && (k + 3) < K) {
                a4 = *((float4*)&A[global_row * K + k]);
            } else {
                a4 = make_float4(
                    (global_row < M && k + 0 < K) ? A[global_row * K + k + 0] : 0.0f,
                    (global_row < M && k + 1 < K) ? A[global_row * K + k + 1] : 0.0f,
                    (global_row < M && k + 2 < K) ? A[global_row * K + k + 2] : 0.0f,
                    (global_row < M && k + 3 < K) ? A[global_row * K + k + 3] : 0.0f
                );
            }
            As[0][ty + mi * THREADS_PER_BLOCK_Y][k + 0] = a4.x;
            As[0][ty + mi * THREADS_PER_BLOCK_Y][k + 1] = a4.y;
            As[0][ty + mi * THREADS_PER_BLOCK_Y][k + 2] = a4.z;
            As[0][ty + mi * THREADS_PER_BLOCK_Y][k + 3] = a4.w;
        }
    }
    
    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
        int global_k = k;
        #pragma unroll
        for (int ni = 0; ni < THREAD_TILE_N; ++ni) {
            int col = b_col_base + tx + ni * THREADS_PER_BLOCK_X;
            if (global_k < K && col < N) {
                Bs[0][k][col - b_col_base] = B[global_k * N + col];
            } else {
                Bs[0][k][col - b_col_base] = 0.0f;
            }
        }
    }
    
    __syncthreads();

    // Main loop with optimized double buffering
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Wait for current tile data to be ready (except first iteration which was prefetched)
        if (tile > 0) {
            __syncthreads();
        }
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int mi = 0; mi < THREAD_TILE_M; ++mi) {
                float a_val = As[curr_buffer][ty + mi * THREADS_PER_BLOCK_Y][k];
                #pragma unroll
                for (int ni = 0; ni < THREAD_TILE_N; ++ni) {
                    c_local[mi][ni] += a_val * Bs[curr_buffer][k][tx + ni * THREADS_PER_BLOCK_X];
                }
            }
        }
        
        // Prepare next tile load (if not last iteration)
        if (tile + 1 < num_tiles) {
            int next_buffer = 1 - curr_buffer;
            int next_k_start = (tile + 1) * TILE_K;
            
            #pragma unroll
            for (int mi = 0; mi < THREAD_TILE_M; ++mi) {
                int global_row = a_row_base + ty + mi * THREADS_PER_BLOCK_Y;
                #pragma unroll
                for (int k = 0; k < TILE_K; k += 4) {
                    float4 a4;
                    int global_k = next_k_start + k;
                    if (global_row < M && global_k + 3 < K) {
                        a4 = *((float4*)&A[global_row * K + global_k]);
                    } else {
                        a4 = make_float4(
                            (global_row < M && global_k + 0 < K) ? A[global_row * K + global_k + 0] : 0.0f,
                            (global_row < M && global_k + 1 < K) ? A[global_row * K + global_k + 1] : 0.0f,
                            (global_row < M && global_k + 2 < K) ? A[global_row * K + global_k + 2] : 0.0f,
                            (global_row < M && global_k + 3 < K) ? A[global_row * K + global_k + 3] : 0.0f
                        );
                    }
                    As[next_buffer][ty + mi * THREADS_PER_BLOCK_Y][k + 0] = a4.x;
                    As[next_buffer][ty + mi * THREADS_PER_BLOCK_Y][k + 1] = a4.y;
                    As[next_buffer][ty + mi * THREADS_PER_BLOCK_Y][k + 2] = a4.z;
                    As[next_buffer][ty + mi * THREADS_PER_BLOCK_Y][k + 3] = a4.w;
                }
            }
            
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                int global_k = next_k_start + k;
                #pragma unroll
                for (int ni = 0; ni < THREAD_TILE_N; ++ni) {
                    int col = b_col_base + tx + ni * THREADS_PER_BLOCK_X;
                    if (global_k < K && col < N) {
                        Bs[next_buffer][k][col - b_col_base] = B[global_k * N + col];
                    } else {
                        Bs[next_buffer][k][col - b_col_base] = 0.0f;
                    }
                }
            }
            
            curr_buffer = next_buffer;
        }
    }
    
    // Final synchronization to ensure all computes are done before writing
    __syncthreads();
    
    // Write results back with accumulation support
    #pragma unroll
    for (int mi = 0; mi < THREAD_TILE_M; ++mi) {
        int row = by * TILE_M + ty + mi * THREADS_PER_BLOCK_Y;
        if (row >= M) continue;
        
        #pragma unroll
        for (int ni = 0; ni < THREAD_TILE_N; ++ni) {
            int col = bx * TILE_N + tx + ni * THREADS_PER_BLOCK_X;
            if (col < N) {
                C[row * N + col] = c_local[mi][ni];
            }
        }
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y); // 32x16 = 512 threads
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    // Calculate shared memory size for double buffering with padding:
    // 2 * (TILE_M * (TILE_K + PAD_A) + TILE_K * TILE_N_PADDED) * sizeof(float)
    size_t shared_mem_size = 2 * (TILE_M * (TILE_K + PAD_A) + TILE_K * TILE_N_PADDED) * sizeof(float);
    
    // Set max dynamic shared memory for compatibility (works on both sm_75 and sm_90)
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    
    // Launch kernel with dynamic shared memory
    gemm_kernel<<<grid, block, shared_mem_size>>>(A, B, C, M, N, K);
}
