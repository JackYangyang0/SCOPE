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
    __shared__ float As[2][BK][BM + 1];
    __shared__ float Bs[2][BK][BN + 1];
    static_assert(BM == 64 && BN == 128 && BK == 16, "SCOPE launch config and shared-memory IR disagree");
    static_assert((BM * BK) % 256 == 0, "A tile load coverage requires BM*BK divisible by thread count");
    static_assert((BN * BK) % 256 == 0, "B tile load coverage requires BN*BK divisible by thread count");
    static_assert(BK % 4 == 0, "A float4 transposed scatter requires BK divisible by 4");
    static_assert(BN % 4 == 0, "B float4 load requires BN divisible by 4");
/*
     * SHARED_DECL_END
     */

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int thread_num = 256;
    const int tile_m0 = blockIdx.y * BM;
    const int tile_n0 = blockIdx.x * BN;

    /*
     * INDEX_MAPPING_BEGIN
     */
    const int warp_tiles_n = BN / WN;
    const int Wrow = wid / warp_tiles_n;
    const int Wcol = wid - Wrow * warp_tiles_n;
    const int lane_cols = WNITER / TN;
    const int Trow = lane / lane_cols;
    const int Tcol = lane - Trow * lane_cols;

    const int load_a_thread_k = tid / (BM / 4);
    const int load_a_thread_m = (tid % (BM / 4)) * 4;
    const int load_b_thread_k = tid / (BN / 4);
    const int load_b_thread_n = (tid % (BN / 4)) * 4;
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

    const int A_base = tile_m0 * K;
    const int B_base = tile_n0;
    const int k_tiles = K / BK;

    /*
     * GLOBAL_TO_SHARED_LOAD_BEGIN
     */
    {
        int a_k = load_a_thread_k;
        int a_m = load_a_thread_m;
        float4 tmp = FLOAT4(A[A_base + a_k + a_m * K]);
        As[0][a_k][a_m] = tmp.x;
        As[0][a_k + 1][a_m] = tmp.y;
        As[0][a_k + 2][a_m] = tmp.z;
        As[0][a_k + 3][a_m] = tmp.w;
    }
    {
        int b_k = load_b_thread_k;
        int b_n = load_b_thread_n;
        FLOAT4(Bs[0][b_k][b_n]) = FLOAT4(B[B_base + b_k * N + b_n]);
    }
/*
     * GLOBAL_TO_SHARED_LOAD_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
        /*
         * SYNC_AFTER_LOAD_BEGIN
         */
        __syncthreads();
        /*
         * SYNC_AFTER_LOAD_END
         */
        const int comp_flag = (bkIdx - 1) & 1;
        const int mem_flag = bkIdx & 1;

        /*
         * COMPUTE_INNER_BEGIN
         */
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WM / WMITER; ++wm) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                }
                #pragma unroll
                for (int wn = 0; wn < WN / WNITER; ++wn) {
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
        {
            int a_k = load_a_thread_k + bkIdx * BK;
            int a_m = load_a_thread_m;
            float4 tmp = FLOAT4(A[A_base + a_k + a_m * K]);
            As[mem_flag][load_a_thread_k][a_m] = tmp.x;
            As[mem_flag][load_a_thread_k + 1][a_m] = tmp.y;
            As[mem_flag][load_a_thread_k + 2][a_m] = tmp.z;
            As[mem_flag][load_a_thread_k + 3][a_m] = tmp.w;
        }
        {
            int b_k = load_b_thread_k + bkIdx * BK;
            int b_n = load_b_thread_n;
            FLOAT4(Bs[mem_flag][load_b_thread_k][b_n]) = FLOAT4(B[B_base + b_k * N + b_n]);
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
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
            }
            #pragma unroll
            for (int wn = 0; wn < WN / WNITER; ++wn) {
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
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    const int global_m = tile_m0 + Wrow * WM + wm * WMITER + Trow * TM + m;
                    const int global_n = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN + n;
                    if (global_m < M && global_n < N) {
                        const int c_index = OFFSET(global_m, global_n, N);
                        C[c_index] = alpha * results[wm * TM + m][wn * TN + n] + beta * C[c_index];
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
    static const int WM = 16;
    static const int WN = 64;
    static const int WMITER = 8;
    static const int WNITER = 32;
    static const int TM = 2;
    static const int TN = 4;
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
