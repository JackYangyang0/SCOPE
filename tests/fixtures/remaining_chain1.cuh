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
    static_assert(BM == 128 && BN == 128 && BK == 32, "SCOPE launch config and shared-memory IR disagree");
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
    const int warp_tiles_n = BN / WN;
    const int Wrow = wid / warp_tiles_n;
    const int Wcol = wid - Wrow * warp_tiles_n;
    const int lane_cols = WNITER / TN;
    const int Trow = lane / lane_cols;
    const int Tcol = lane - Trow * lane_cols;
    const int local_m_base = Wrow * WM + Trow * TM;
    const int local_n_base = Wcol * WN + Tcol * TN;
    const int load_a_smem_m = tid % BM;
    const int load_a_smem_k = (tid / BM) % BK;
    const int load_b_smem_n = tid % BN;
    const int load_b_smem_k = (tid / BN) % BK;
    const int hightA = CEIL_DIV(BM * BK, thread_num);
    const int hightB = CEIL_DIV(BN * BK, thread_num);
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
    for (int loadOffset = 0; loadOffset < hightA; ++loadOffset) {
        const int a_m = load_a_smem_m + loadOffset * (thread_num / BK);
        const int a_k = load_a_smem_k;
        if (a_m < BM) {
            const int gm = tile_m0 + a_m;
            const int gk = a_k;
            if (gm < M && gk < K) {
                const float4 tmp = FLOAT4(A[OFFSET(gm, gk, K)]);
                As[0][a_k + 0][a_m] = tmp.x;
                As[0][a_k + 1][a_m] = tmp.y;
                As[0][a_k + 2][a_m] = tmp.z;
                As[0][a_k + 3][a_m] = tmp.w;
            } else {
                for (int kk = 0; kk < 4; ++kk) {
                    const int gk_s = a_k + kk;
                    As[0][gk_s][a_m] = (gm < M && gk_s < K) ? A[OFFSET(gm, gk_s, K)] : 0.0f;
                }
            }
        }
    }
    for (int loadOffset = 0; loadOffset < hightB; ++loadOffset) {
        const int b_k = load_b_smem_k + loadOffset * (thread_num / BN);
        const int b_n = load_b_smem_n;
        if (b_k < BK) {
            const int gk = b_k;
            const int gn = tile_n0 + b_n;
            if (gk < K && gn + 3 < N) {
                const float4 bv = FLOAT4(B[OFFSET(gk, gn, N)]);
                Bs[0][b_k][b_n + 0] = bv.x;
                Bs[0][b_k][b_n + 1] = bv.y;
                Bs[0][b_k][b_n + 2] = bv.z;
                Bs[0][b_k][b_n + 3] = bv.w;
            } else {
                for (int nn = 0; nn < 4; ++nn) {
                    const int gn_s = tile_n0 + b_n + nn;
                    Bs[0][b_k][b_n + nn] = (gk < K && gn_s < N) ? B[OFFSET(gk, gn_s, N)] : 0.0f;
                }
            }
        }
    }
/*
     * GLOBAL_TO_SHARED_LOAD_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
        __syncthreads();
        const int comp_flag = (bkIdx - 1) & 1;
        const int mem_flag = bkIdx & 1;
        const int k_base = bkIdx * BK;

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

        for (int loadOffset = 0; loadOffset < hightA; ++loadOffset) {
            const int a_m = load_a_smem_m + loadOffset * (thread_num / BK);
            const int a_k = load_a_smem_k;
            if (a_m < BM) {
                const int gm = tile_m0 + a_m;
                const int gk = k_base + a_k;
                if (gm < M && gk + 3 < K) {
                    const float4 tmp = FLOAT4(A[OFFSET(gm, gk, K)]);
                    As[mem_flag][a_k + 0][a_m] = tmp.x;
                    As[mem_flag][a_k + 1][a_m] = tmp.y;
                    As[mem_flag][a_k + 2][a_m] = tmp.z;
                    As[mem_flag][a_k + 3][a_m] = tmp.w;
                } else {
                    for (int kk = 0; kk < 4; ++kk) {
                        const int gk_s = k_base + a_k + kk;
                        As[mem_flag][a_k + kk][a_m] = (gm < M && gk_s < K) ? A[OFFSET(gm, gk_s, K)] : 0.0f;
                    }
                }
            }
        }
        for (int loadOffset = 0; loadOffset < hightB; ++loadOffset) {
            const int b_k = load_b_smem_k + loadOffset * (thread_num / BN);
            const int b_n = load_b_smem_n;
            if (b_k < BK) {
                const int gk = k_base + b_k;
                const int gn = tile_n0 + b_n;
                if (gk < K && gn + 3 < N) {
                    const float4 bv = FLOAT4(B[OFFSET(gk, gn, N)]);
                    Bs[mem_flag][b_k][b_n + 0] = bv.x;
                    Bs[mem_flag][b_k][b_n + 1] = bv.y;
                    Bs[mem_flag][b_k][b_n + 2] = bv.z;
                    Bs[mem_flag][b_k][b_n + 3] = bv.w;
                } else {
                    for (int nn = 0; nn < 4; ++nn) {
                        const int gn_s = tile_n0 + b_n + nn;
                        Bs[mem_flag][b_k][b_n + nn] = (gk < K && gn_s < N) ? B[OFFSET(gk, gn_s, N)] : 0.0f;
                    }
                }
            }
        }
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
    static const int BK = 32;
    static const int WM = 32;
    static const int WN = 32;
    static const int WMITER = 16;
    static const int WNITER = 32;
    static const int TM = 4;
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
