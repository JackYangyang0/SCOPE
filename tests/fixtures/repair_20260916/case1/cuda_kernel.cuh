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
    __shared__ float As[2][BM][BK + 1];
    __shared__ float Bs[2][BK][BN + 1];
    static_assert(BM == 128 && BN == 128 && BK == 16, "SCOPE launch config and shared-memory IR disagree");
/*
     * SHARED_DECL_END
     */

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
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
    const int local_m_base = Wrow * WM + Trow * TM;
    const int local_n_base = Wcol * WN + Tcol * TN;
    const int global_m_base = tile_m0 + local_m_base;
    const int global_n_base = tile_n0 + local_n_base;
    (void)global_m_base;
    (void)global_n_base;
    const int load_a_smem_m = tid / (BK / 4);
    const int load_a_smem_k = (tid % (BK / 4)) * 4;
    const int load_b_smem_n = (tid % (BN / 4)) * 4;
    const int load_b_smem_k = tid / (BN / 4);
    const int load_a_steps = (BM * BK) / (BK * 4);
    const int load_b_steps = (BK * BN) / (BN * 4);
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

    const int k_tiles = K / BK;

    /*
     * GLOBAL_TO_SHARED_LOAD_BEGIN
     */
    for (int step = 0; step < load_a_steps; ++step) {
        int local_m = load_a_smem_m + step * (BK / 4);
        int local_k = load_a_smem_k;
        int gm = tile_m0 + local_m;
        int gk = local_k;
        float4 tmp = FLOAT4(A[gm * K + gk]);
        As[0][local_m][local_k]     = tmp.x;
        As[0][local_m][local_k + 1] = tmp.y;
        As[0][local_m][local_k + 2] = tmp.z;
        As[0][local_m][local_k + 3] = tmp.w;
    }
    for (int step = 0; step < load_b_steps; ++step) {
        int local_k = load_b_smem_k + step * (BN / 4);
        int local_n = load_b_smem_n;
        int gk = local_k;
        int gn = tile_n0 + local_n;
        float4 tmp = FLOAT4(B[gk * N + gn]);
        Bs[0][local_k][local_n]     = tmp.x;
        Bs[0][local_k][local_n + 1] = tmp.y;
        Bs[0][local_k][local_n + 2] = tmp.z;
        Bs[0][local_k][local_n + 3] = tmp.w;
    }
    __syncthreads();
/*
     * GLOBAL_TO_SHARED_LOAD_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    for (int bkIdx = 0; bkIdx < k_tiles; ++bkIdx) {
        const int comp_flag = bkIdx & 1;
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WM / WMITER; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WN / WNITER; ++wn) {
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        regM[i] = As[comp_flag][Wrow * WM + wm * WMITER + Trow * TM + i][k];
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
        if (bkIdx < k_tiles - 1) {
            const int mem_flag = (bkIdx + 1) & 1;
            for (int step = 0; step < load_a_steps; ++step) {
                int local_m = load_a_smem_m + step * (BK / 4);
                int local_k = load_a_smem_k;
                int gm = tile_m0 + local_m;
                int gk = (bkIdx + 1) * BK + local_k;
                float4 tmp = FLOAT4(A[gm * K + gk]);
                As[mem_flag][local_m][local_k]     = tmp.x;
                As[mem_flag][local_m][local_k + 1] = tmp.y;
                As[mem_flag][local_m][local_k + 2] = tmp.z;
                As[mem_flag][local_m][local_k + 3] = tmp.w;
            }
            for (int step = 0; step < load_b_steps; ++step) {
                int local_k = load_b_smem_k + step * (BN / 4);
                int local_n = load_b_smem_n;
                int gk = (bkIdx + 1) * BK + local_k;
                int gn = tile_n0 + local_n;
                float4 tmp = FLOAT4(B[gk * N + gn]);
                Bs[mem_flag][local_k][local_n]     = tmp.x;
                Bs[mem_flag][local_k][local_n + 1] = tmp.y;
                Bs[mem_flag][local_k][local_n + 2] = tmp.z;
                Bs[mem_flag][local_k][local_n + 3] = tmp.w;
            }
            __syncthreads();
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
                        C[c_index] = alpha * results[m + wm * TM][n + wn * TN] + beta * C[c_index];
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
    static const int BM = 128;
    static const int BN = 128;
    static const int BK = 16;
    static const int WM = 16;
    static const int WN = 64;
    static const int WMITER = 4;
    static const int WNITER = 32;
    static const int TM = 2;
    static const int TN = 2;
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
