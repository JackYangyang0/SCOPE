#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block and thread indices
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // Shared memory for tiles of A and B with double buffering
    extern __shared__ float shared_mem[];
    float* As[2] = {shared_mem, shared_mem + 32 * 32};
    float* Bs[2] = {shared_mem + 2 * 32 * 32, shared_mem + 3 * 32 * 32};

    // Register accumulators for output tile (2x2 per thread)
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // Global memory pointers for current tile
    const float* Ap = A + block_row * 32 * K;
    const float* Bp = B + block_col * 32;

    // Loop over tiles
    const int num_tiles = (K + 31) / 32;
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Load next tile into shared memory (double buffering)
        const int next_tile = tile + 1;
        const bool is_valid_tile = (next_tile * 32 < K);
        const float* Ap_next = Ap + 32 * K;
        const float* Bp_next = Bp + 32 * K;

        // Cooperative loading of A tile (32x32)
        for (int load_idx = thread_id; load_idx < 512; load_idx += blockDim.x * blockDim.y) {
            const int load_row = load_idx / 16;
            const int load_col = (load_idx % 16) * 2;
            const int global_row = block_row * 32 + load_row;
            const int global_col = tile * 32 + load_col;
            
            float2 a_val = make_float2(0.0f, 0.0f);
            if (global_row < M && global_col + 1 < K) {
                a_val = *reinterpret_cast<const float2*>(&A[global_row * K + global_col]);
            } else if (global_row < M && global_col < K) {
                a_val.x = A[global_row * K + global_col];
                a_val.y = 0.0f;
            }
            *reinterpret_cast<float2*>(&As[tile & 1][load_row * 32 + load_col]) = a_val;
        }

        // Cooperative loading of B tile (32x32)
        for (int load_idx = thread_id; load_idx < 512; load_idx += blockDim.x * blockDim.y) {
            const int load_row = load_idx / 16;
            const int load_col = (load_idx % 16) * 2;
            const int global_row = tile * 32 + load_row;
            const int global_col = block_col * 32 + load_col;
            
            float2 b_val = make_float2(0.0f, 0.0f);
            if (global_row < K && global_col + 1 < N) {
                b_val = *reinterpret_cast<const float2*>(&B[global_row * N + global_col]);
            } else if (global_row < K && global_col < N) {
                b_val.x = B[global_row * N + global_col];
                b_val.y = 0.0f;
            }
            *reinterpret_cast<float2*>(&Bs[tile & 1][load_row * 32 + load_col]) = b_val;
        }

        __syncthreads();

        // Compute partial dot product for current tile
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float a0 = As[tile & 1][(threadIdx.y * 2) * 32 + k];
            const float a1 = As[tile & 1][(threadIdx.y * 2 + 1) * 32 + k];
            const float b0 = Bs[tile & 1][k * 32 + threadIdx.x * 2];
            const float b1 = Bs[tile & 1][k * 32 + threadIdx.x * 2 + 1];
            
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }

        __syncthreads();

        // Update pointers for next iteration
        Ap = Ap_next;
        Bp = Bp_next;
    }

    // Write results to global memory
    const int c_row_base = block_row * 32 + threadIdx.y * 2;
    const int c_col_base = block_col * 32 + threadIdx.x * 2;

    if (c_row_base < M && c_col_base < N) {
        C[c_row_base * N + c_col_base] = acc[0][0];
    }
    if (c_row_base < M && c_col_base + 1 < N) {
        C[c_row_base * N + c_col_base + 1] = acc[0][1];
    }
    if (c_row_base + 1 < M && c_col_base < N) {
        C[(c_row_base + 1) * N + c_col_base] = acc[1][0];
    }
    if (c_row_base + 1 < M && c_col_base + 1 < N) {
        C[(c_row_base + 1) * N + c_col_base + 1] = acc[1][1];
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    // Configure kernel launch parameters
    const dim3 block_size(16, 16);  // 256 threads per block
    const dim3 grid_size((N + 31) / 32, (M + 31) / 32);
    
    // Shared memory size: 4 tiles of 32x32 floats = 4 * 1024 * sizeof(float) = 16384 bytes
    const size_t shared_mem_size = 4 * 32 * 32 * sizeof(float);
    
    // Launch kernel
    gemm_kernel<<<grid_size, block_size, shared_mem_size>>>(A, B, C, M, N, K);
}
