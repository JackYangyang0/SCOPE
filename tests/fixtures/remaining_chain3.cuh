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

    const int num_threads = blockDim.x;
    const int a_vec_count = BM * (BK / 4);
    const int b_vec_count = BK * (BN / 4);
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
    {
        int v = tid;
        if (v < a_vec_count) {
            int am = v / (BK / 4);
            int ak = (v % (BK / 4)) * 4;
            int global_m = tile_m0 + am;
            int global_k = ak;
            if (global_m < M && global_k + 3 < K) {
                float4 tmp = FLOAT4(A[OFFSET(global_m, global_k, K)]);
                As[0][ak + 0][am] = tmp.x;
                As[0][ak + 1][am] = tmp.y;
                As[0][ak + 2][am] = tmp.z;
                As[0][ak + 3][am] = tmp.w;
            } else {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    int gk = global_k + kk;
                    As[0][ak + kk][am] = (global_m < M && gk < K) ? A[OFFSET(global_m, gk, K)] : 0.0f;
                }
            }
        }
    }
    {
        int v = tid;
        if (v < b_vec_count) {
            int bk = v / (BN / 4);
            int bn = (v % (BN / 4)) * 4;
            int global_k = bk;
            int global_n = tile_n0 + bn;
            if (global_k < K && global_n + 3 < N) {
                float4 tmp = FLOAT4(B[OFFSET(global_k, global_n, N)]);
                Bs[0][bk][bn + 0] = tmp.x;
                Bs[0][bk][bn + 1] = tmp.y;
                Bs[0][bk][bn + 2] = tmp.z;
                Bs[0][bk][bn + 3] = tmp.w;
            } else {
                #pragma unroll
                for (int nn = 0; nn < 4; ++nn) {
                    int gn = global_n + nn;
                    Bs[0][bk][bn + nn] = (global_k < K && gn < N) ? B[OFFSET(global_k, gn, N)] : 0.0f;
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
        {
            int v = tid;
            if (v < a_vec_count) {
                int am = v / (BK / 4);
                int ak = (v % (BK / 4)) * 4;
                int global_m = tile_m0 + am;
                int global_k = bkIdx * BK + ak;
                if (global_m < M && global_k + 3 < K) {
                    float4 tmp = FLOAT4(A[OFFSET(global_m, global_k, K)]);
                    As[mem_flag][ak + 0][am] = tmp.x;
                    As[mem_flag][ak + 1][am] = tmp.y;
                    As[mem_flag][ak + 2][am] = tmp.z;
                    As[mem_flag][ak + 3][am] = tmp.w;
                } else {
                    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        int gk = global_k + kk;
                        As[mem_flag][ak + kk][am] = (global_m < M && gk < K) ? A[OFFSET(global_m, gk, K)] : 0.0f;
                    }
                }
            }
        }
        {
            int v = tid;
            if (v < b_vec_count) {
                int bk = v / (BN / 4);
                int bn = (v % (BN / 4)) * 4;
                int global_k = bkIdx * BK + bk;
                int global_n = tile_n0 + bn;
                if (global_k < K && global_n + 3 < N) {
                    float4 tmp = FLOAT4(B[OFFSET(global_k, global_n, N)]);
                    Bs[mem_flag][bk][bn + 0] = tmp.x;
                    Bs[mem_flag][bk][bn + 1] = tmp.y;
                    Bs[mem_flag][bk][bn + 2] = tmp.z;
                    Bs[mem_flag][bk][bn + 3] = tmp.w;
                } else {
                    #pragma unroll
                    for (int nn = 0; nn < 4; ++nn) {
                        int gn = global_n + nn;
                        Bs[mem_flag][bk][bn + nn] = (global_k < K && gn < N) ? B[OFFSET(global_k, gn, N)] : 0.0f;
                    }
                }
            }
        }
/*
         * NEXT_TILE_LOAD_END
         */
    }

    __syncthreads();
    {
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
