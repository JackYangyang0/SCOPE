#pragma once

#include <cuda_runtime.h>

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

#ifndef OFFSET
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif

#ifndef FLOAT4
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#endif

template <
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int WMITER,
    int WNITER,
    int TM,
    int TN>
__global__ void gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {
    /*
     * SHARED_DECL_BEGIN
     */
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];
    static_assert(BM == 64 && BN == 128 && BK == 16, "SCOPE launch config and shared-memory IR disagree");
/*
     * SHARED_DECL_END
     */

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int thread_num = BM * BN / WM / WN * 32;
    const int tile_m0 = blockIdx.y * BM;
    const int tile_n0 = blockIdx.x * BN;

    /*
     * INDEX_MAPPING_BEGIN
     */
    const int elements_per_tile = BM * BN;
    const int warp_tiles_n = BN / WN;
    const int Wrow = wid / warp_tiles_n;
    const int Wcol = wid - Wrow * warp_tiles_n;
    const int lane_cols = WNITER / TN;
    const int Trow = lane / lane_cols;
    const int Tcol = lane - Trow * lane_cols;
    const int local_m_base = Wrow * WM + Trow * TM;
    const int local_n_base = Wcol * WN + Tcol * TN;
    const int global_m_base = tile_m0 + local_m_base;
    const int global_n_base = tile_n0 + local_n_base;
    const int global_m = global_m_base;
    const int global_n = global_n_base;
    // Cooperative float4 load mapping: each thread loads 4 consecutive elements
    // A: 256 threads cover BM*BK/4 = 64*16/4 = 256 vectors
    const int load_a_smem_m = tid / (BK / 4);
    const int load_a_smem_k_offset = (tid % (BK / 4)) * 4;
    // B: 256 threads cover BK*BN/4 = 16*128/4 = 512 vectors (2 per thread)
    const int b_vectors_per_thread = (BK * BN / 4) / thread_num;
    const int load_b_smem_n_offset = (tid % (BN / 4)) * 4;
    const int load_b_smem_k_stride = tid / (BN / 4);
    (void)global_m_base;
    (void)global_n_base;
/*
     * INDEX_MAPPING_END
     */

    /*
     * REGISTER_DECL_BEGIN
     */
    float results[WM / WMITER * TM][WN / WNITER * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};
    /*
     * REGISTER_DECL_END
     */

    const int A_offset = tile_m0 * K;
    const int B_offset = tile_n0;
    const int C_offset = tile_m0 * N + tile_n0;
    const int k_tiles = K / BK;

    /*
     * GLOBAL_TO_SHARED_LOAD_BEGIN
     */
    const int k_base = 0;
    const int stage = 0;
    // Load A tile: float4 vector load with scalar scatter to shared memory
    {
        const int global_m = tile_m0 + load_a_smem_m;
        const int global_k = k_base + load_a_smem_k_offset;
        const int a_index = global_m * K + global_k;
        float4 tmp = FLOAT4(A[a_index]);
        As[stage][load_a_smem_k_offset + 0][load_a_smem_m] = tmp.x;
        As[stage][load_a_smem_k_offset + 1][load_a_smem_m] = tmp.y;
        As[stage][load_a_smem_k_offset + 2][load_a_smem_m] = tmp.z;
        As[stage][load_a_smem_k_offset + 3][load_a_smem_m] = tmp.w;
    }
    // Load B tile: each thread loads b_vectors_per_thread float4 vectors
    for (int b_vec = 0; b_vec < b_vectors_per_thread; ++b_vec) {
        const int load_b_smem_k = load_b_smem_k_stride + b_vec * (thread_num / (BN / 4));
        const int load_b_smem_n = load_b_smem_n_offset;
        const int global_k = k_base + load_b_smem_k;
        const int global_n = tile_n0 + load_b_smem_n;
        const int b_index = global_k * N + global_n;
        float4 tmp = FLOAT4(B[b_index]);
        Bs[stage][load_b_smem_k][load_b_smem_n + 0] = tmp.x;
        Bs[stage][load_b_smem_k][load_b_smem_n + 1] = tmp.y;
        Bs[stage][load_b_smem_k][load_b_smem_n + 2] = tmp.z;
        Bs[stage][load_b_smem_k][load_b_smem_n + 3] = tmp.w;
    }
    __syncthreads();
/*
     * GLOBAL_TO_SHARED_LOAD_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
        const int comp_flag = (bkIdx - 1) & 1;
        const int mem_flag = bkIdx & 1;
        const int k_base = bkIdx * BK;

        /*
         * SYNC_AFTER_LOAD_BEGIN
         */
        __syncthreads();
        /*
         * SYNC_AFTER_LOAD_END
         */

        /*
         * COMPUTE_INNER_BEGIN
         */
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WM / WMITER; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WN / WNITER; ++wn) {
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                    }
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        regN[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
                    }
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];
                        }
                    }
                }
            }
        }
        /*
         * COMPUTE_INNER_END
         */

        /*
         * NEXT_TILE_LOAD_BEGIN
         */
        // Load next A tile: reuse precomputed indices, update k_base and stage
        {
            const int global_m = tile_m0 + load_a_smem_m;
            const int global_k = k_base + load_a_smem_k_offset;
            const int a_index = global_m * K + global_k;
            float4 tmp = FLOAT4(A[a_index]);
            As[mem_flag][load_a_smem_k_offset + 0][load_a_smem_m] = tmp.x;
            As[mem_flag][load_a_smem_k_offset + 1][load_a_smem_m] = tmp.y;
            As[mem_flag][load_a_smem_k_offset + 2][load_a_smem_m] = tmp.z;
            As[mem_flag][load_a_smem_k_offset + 3][load_a_smem_m] = tmp.w;
        }
        // Load next B tile: reuse precomputed indices with updated k_base
        for (int b_vec = 0; b_vec < b_vectors_per_thread; ++b_vec) {
            const int load_b_smem_k = load_b_smem_k_stride + b_vec * (thread_num / (BN / 4));
            const int load_b_smem_n = load_b_smem_n_offset;
            const int global_k = k_base + load_b_smem_k;
            const int global_n = tile_n0 + load_b_smem_n;
            const int b_index = global_k * N + global_n;
            float4 tmp = FLOAT4(B[b_index]);
            Bs[mem_flag][load_b_smem_k][load_b_smem_n + 0] = tmp.x;
            Bs[mem_flag][load_b_smem_k][load_b_smem_n + 1] = tmp.y;
            Bs[mem_flag][load_b_smem_k][load_b_smem_n + 2] = tmp.z;
            Bs[mem_flag][load_b_smem_k][load_b_smem_n + 3] = tmp.w;
        }
/*
         * NEXT_TILE_LOAD_END
         */
    }

    __syncthreads();
    const int comp_flag = (k_tiles - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int wm = 0; wm < WM / WMITER; ++wm) {
            #pragma unroll
            for (int wn = 0; wn < WN / WNITER; ++wn) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    regN[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];
                    }
                }
            }
        }
    }
/*
     * MAIN_LOOP_END
     */

    /*
     * STORE_BEGIN
     */
    #pragma unroll
    for (int wm = 0; wm < WM / WMITER; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WN / WNITER; ++wn) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const int global_m = tile_m0 + Wrow * WM + wm * WMITER + Trow * TM + m;
                const int global_n_base = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN;
                if (global_m < M && global_n_base + 3 < N) {
                    float4 out;
                    out.x = alpha * results[wm * TM + m][wn * TN + 0];
                    out.y = alpha * results[wm * TM + m][wn * TN + 1];
                    out.z = alpha * results[wm * TM + m][wn * TN + 2];
                    out.w = alpha * results[wm * TM + m][wn * TN + 3];
                    *reinterpret_cast<float4*>(&C[global_m * N + global_n_base]) = out;
                } else {
                    #pragma unroll
                    for (int n = 0; n < TN; ++n) {
                        const int global_n = global_n_base + n;
                        if (global_m < M && global_n < N) {
                            C[global_m * N + global_n] = alpha * results[wm * TM + m][wn * TN + n];
                        }
                    }
                }
            }
        }
    }
/*
     * STORE_END
     */
}

void cuda_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {
    /*
     * LAUNCH_CONFIG_BEGIN
     */
    static const int BM = 64;
    static const int BN = 128;
    static const int BK = 16;
    static const int WM = 32;
    static const int WN = 32;
    static const int WMITER = 16;
    static const int WNITER = 32;
    static const int TM = 4;
    static const int TN = 4;
    const bool divisible = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);
    const bool aligned_A = (reinterpret_cast<uintptr_t>(A) % 16 == 0) && (K % 4 == 0);
    const bool aligned_B = (reinterpret_cast<uintptr_t>(B) % 16 == 0) && (N % 4 == 0);
    const bool aligned_C = (reinterpret_cast<uintptr_t>(C) % 16 == 0) && (N % 4 == 0);
    if (!divisible || !aligned_A || !aligned_B || !aligned_C) {
        return;
    }
    dim3 block((BM * BN) / (WM * WN) * 32, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
/*
     * LAUNCH_CONFIG_END
     */

    dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);
}
