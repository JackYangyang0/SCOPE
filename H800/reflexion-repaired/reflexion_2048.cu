#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Optimized for both sm_75 (RTX 2060) and sm_90 (H800) with careful tradeoffs
// Uses FP16 WMMA which works on both architectures
// Shared memory sized to fit within 64KB dynamic limit with minimal padding for bank conflict avoidance
#define TILE_M 128
#define TILE_N 128          // Reduced from 256 to fit within 48KB static shared memory limit on sm_75
#define TILE_K 64
#define THREADS_PER_BLOCK 256  // 8 warps: 2 in M, 4 in N (128/64=2, 128/32=4)
#define WARP_SIZE 32
#define WARPS_M 2
#define WARPS_N 4

// Each warp handles a 64x32 output tile, divided into 4x2 WMMA tiles (16x16 each)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// WMMA half fragments require a leading dimension compatible with 16-byte
// transactions. A stride of 65 is invalid even though it avoids bank conflicts.
#define SMEM_PAD_A 0
#define SMEM_PAD_B 0

__global__ void convert_kernel(const float* input, half* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = __float2half_rn(input[idx]);
        }
    }

__global__ void gemm_kernel(const half* __restrict__ A_half, const half* __restrict__ B_half, float* __restrict__ C, int M, int N, int K) {
    // Calculate shared memory layout with padding
    constexpr int smem_a_size = TILE_M * (TILE_K + SMEM_PAD_A);
    constexpr int smem_b_size = TILE_N * (TILE_K + SMEM_PAD_B);
    
    extern __shared__ half shared_mem[];
    
    // Double buffering pointers
    half* smem_a0 = shared_mem;
    half* smem_b0 = smem_a0 + smem_a_size;
    half* smem_a1 = smem_b0 + smem_b_size;
    half* smem_b1 = smem_a1 + smem_a_size;
    
    half* smem_a[2] = {smem_a0, smem_a1};
    half* smem_b[2] = {smem_b0, smem_b1};

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Warp's position within the block tile
    int warp_row = warp_id / WARPS_N;  // 0 or 1
    int warp_col = warp_id % WARPS_N;  // 0-3
    
    // This warp handles a 64x32 output tile (WMMA_M * 4 x WMMA_N * 2)
    int warp_tile_m = block_row * TILE_M + warp_row * 64;
    int warp_tile_n = block_col * TILE_N + warp_col * 32;
    
    // Accumulators: 4x2 WMMA tiles per warp (64x32 total)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4][2];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
    
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    // Load first tile into shared memory using vectorized half2 loads
    {
        int k_start = 0;
        const int total_threads = blockDim.x;
        
        // Load A tile into the padded row-major shared layout.
        for (int idx = threadIdx.x; idx < TILE_M * TILE_K; idx += total_threads) {
            int m_local = idx / TILE_K;
            int k_local = idx % TILE_K;
            int global_m = block_row * TILE_M + m_local;
            int global_k = k_start + k_local;
            smem_a[0][m_local * (TILE_K + SMEM_PAD_A) + k_local] =
                (global_m < M && global_k < K) ? A_half[global_m * K + global_k] : __float2half(0.0f);
        }
        
        // Load B tile: B is ROW-MAJOR (K x N), so element (k, n) is at B_half[k * N + n]
        for (int idx = threadIdx.x; idx < TILE_N * TILE_K; idx += total_threads) {
            int k_local = idx / TILE_N;
            int n_local = idx % TILE_N;
            int global_n = block_col * TILE_N + n_local;
            int global_k = k_start + k_local;
            smem_b[0][n_local * (TILE_K + SMEM_PAD_B) + k_local] =
                (global_n < N && global_k < K) ? B_half[global_k * N + global_n] : __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Main GEMM loop with double buffering
    for (int tile = 0; tile < num_k_tiles; tile++) {
        int buffer_idx = tile % 2;
        int next_buffer = 1 - buffer_idx;
        int k_start_next = (tile + 1) * TILE_K;
        
        // Compute using current tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k += WMMA_K) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    // Load A fragment (row-major) from M-major shared memory
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    int a_row_start = warp_row * 64 + i * WMMA_M;
                    int a_offset = a_row_start * (TILE_K + SMEM_PAD_A) + k;
                    wmma::load_matrix_sync(a_frag, &smem_a[buffer_idx][a_offset], TILE_K + SMEM_PAD_A);
                    
                    // Load B fragment (col-major) from K-major shared memory
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                    int b_col_start = warp_col * 32 + j * WMMA_N;
                    int b_offset = b_col_start * (TILE_K + SMEM_PAD_B) + k;
                    wmma::load_matrix_sync(b_frag, &smem_b[buffer_idx][b_offset], TILE_K + SMEM_PAD_B);
                    
                    // Matrix multiply accumulate
                    wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
                }
            }
        }
        
        // Prefetch next tile if not last iteration
        if (tile < num_k_tiles - 1) {
            const int total_threads = blockDim.x;
            
            for (int idx = threadIdx.x; idx < TILE_M * TILE_K; idx += total_threads) {
                int m_local = idx / TILE_K;
                int k_local = idx % TILE_K;
                int global_m = block_row * TILE_M + m_local;
                int global_k = k_start_next + k_local;
                smem_a[next_buffer][m_local * (TILE_K + SMEM_PAD_A) + k_local] =
                    (global_m < M && global_k < K) ? A_half[global_m * K + global_k] : __float2half(0.0f);
            }
            
            // Load next B tile with coalesced access for ROW-MAJOR input (K x N)
            for (int idx = threadIdx.x; idx < TILE_N * TILE_K; idx += total_threads) {
                int k_local = idx / TILE_N;
                int n_local = idx % TILE_N;
                int global_n = block_col * TILE_N + n_local;
                int global_k = k_start_next + k_local;
                smem_b[next_buffer][n_local * (TILE_K + SMEM_PAD_B) + k_local] =
                    (global_n < N && global_k < K) ? B_half[global_k * N + global_n] : __float2half(0.0f);
            }
        }
        
        __syncthreads();
    }
    
    // Store results to global memory with proper bounds checking
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int out_m = warp_tile_m + i * WMMA_M;
            int out_n = warp_tile_n + j * WMMA_N;
            
            // Check bounds for the entire 16x16 tile
            if (out_m + WMMA_M <= M && out_n + WMMA_N <= N) {
                // Fast path: no bounds checking needed within tile
                wmma::store_matrix_sync(&C[out_m * N + out_n], acc[i][j], N, wmma::mem_row_major);
            } else {
                // Slow path: check each element
                float* C_tile = &C[out_m * N + out_n];
                for (int ti = 0; ti < WMMA_M; ti++) {
                    for (int tj = 0; tj < WMMA_N; tj++) {
                        int actual_m = out_m + ti;
                        int actual_n = out_n + tj;
                        if (actual_m < M && actual_n < N) {
                            C_tile[ti * N + tj] = acc[i][j].x[ti * WMMA_N + tj];
                        }
                    }
                }
            }
        }
    }
}

void gemm_gpu(int M, int N, int K, const float* A_float, const float* B_float, float* C) {
    // Allocate device memory for half precision matrices
    half* A_half;
    half* B_half;
    cudaMalloc(&A_half, M * K * sizeof(half));
    cudaMalloc(&B_half, K * N * sizeof(half));
    
    // Convert A on GPU
    int num_elements_A = M * K;
    int num_blocks_A = (num_elements_A + 255) / 256;
    
    convert_kernel<<<num_blocks_A, 256>>>(A_float, A_half, num_elements_A);
    
    // B is already row-major K x N, matching the benchmark contract.
    int num_elements_B = K * N;
    int num_blocks_B = (num_elements_B + 255) / 256;
    convert_kernel<<<num_blocks_B, 256>>>(B_float, B_half, num_elements_B);
    
    // Configure kernel launch parameters
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 blockDim(THREADS_PER_BLOCK);
    
    // Calculate shared memory requirement
    constexpr int smem_a_size = TILE_M * (TILE_K + SMEM_PAD_A);
    constexpr int smem_b_size = TILE_N * (TILE_K + SMEM_PAD_B);
    size_t total_smem = 2 * (smem_a_size + smem_b_size) * sizeof(half); // ~66KB
    
    // Set dynamic shared memory size (required for >48KB)
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, total_smem);
    
    // Launch kernel
    gemm_kernel<<<gridDim, blockDim, total_smem>>>(A_half, B_half, C, M, N, K);
    
    // Clean up temporary half arrays
    cudaFree(A_half);
    cudaFree(B_half);
}
