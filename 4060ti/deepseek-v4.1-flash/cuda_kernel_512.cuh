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
const int linear_tid = tid;
const int threads = thread_num;
const int a_vec_per_row = BK / 4;
const int a_vec_total = BM * a_vec_per_row;
const int b_vec_per_row = BN / 4;
const int b_vec_total = BK * b_vec_per_row;
const int a_vec_iters = a_vec_total / threads;
const int b_vec_iters = b_vec_total / threads;
static_assert(a_vec_total % threads == 0, "A float4 cooperative load requires complete vector coverage");
static_assert(b_vec_total % threads == 0, "B float4 cooperative load requires complete vector coverage");
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
#pragma unroll
for (int i = 0; i < a_vec_iters; ++i) {
    const int v = linear_tid + i * threads;
    const int am = v / a_vec_per_row;
    const int ak = (v % a_vec_per_row) * 4;
    const int gm = tile_m0 + am;
    const int gk = ak;
    const float4 av = FLOAT4(A[OFFSET(gm, gk, K)]);
    As[0][ak + 0][am] = av.x;
    As[0][ak + 1][am] = av.y;
    As[0][ak + 2][am] = av.z;
    As[0][ak + 3][am] = av.w;
}
#pragma unroll
for (int i = 0; i < b_vec_iters; ++i) {
    const int v = linear_tid + i * threads;
    const int bk = v / b_vec_per_row;
    const int bn = (v % b_vec_per_row) * 4;
    const int gk = bk;
    const int gn = tile_n0 + bn;
    const float4 bv = FLOAT4(B[OFFSET(gk, gn, N)]);
    Bs[0][bk][bn + 0] = bv.x;
    Bs[0][bk][bn + 1] = bv.y;
    Bs[0][bk][bn + 2] = bv.z;
    Bs[0][bk][bn + 3] = bv.w;
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

    /*
     * COMPUTE_INNER_BEGIN
     */
#pragma unroll 16
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
#pragma unroll
for (int i = 0; i < a_vec_iters; ++i) {
    const int v = linear_tid + i * threads;
    const int am = v / a_vec_per_row;
    const int ak = (v % a_vec_per_row) * 4;
    const int gm = tile_m0 + am;
    const int gk = bkIdx * BK + ak;
    const float4 av = FLOAT4(A[OFFSET(gm, gk, K)]);
    As[mem_flag][ak + 0][am] = av.x;
    As[mem_flag][ak + 1][am] = av.y;
    As[mem_flag][ak + 2][am] = av.z;
    As[mem_flag][ak + 3][am] = av.w;
}
#pragma unroll
for (int i = 0; i < b_vec_iters; ++i) {
    const int v = linear_tid + i * threads;
    const int bk = v / b_vec_per_row;
    const int bn = (v % b_vec_per_row) * 4;
    const int gk = bkIdx * BK + bk;
    const int gn = tile_n0 + bn;
    const float4 bv = FLOAT4(B[OFFSET(gk, gn, N)]);
    Bs[mem_flag][bk][bn + 0] = bv.x;
    Bs[mem_flag][bk][bn + 1] = bv.y;
    Bs[mem_flag][bk][bn + 2] = bv.z;
    Bs[mem_flag][bk][bn + 3] = bv.w;
}
/*
     * NEXT_TILE_LOAD_END
     */

    /*
     * SYNC_AFTER_LOAD_BEGIN
     */
    __syncthreads();
    /*
     * SYNC_AFTER_LOAD_END
     */
}

const int comp_flag = (k_tiles - 1) & 1;
#pragma unroll 16
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
            const int global_n = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN;
            if (global_m < M && global_n + (TN - 1) < N) {
                const int c_index = OFFSET(global_m, global_n, N);
                if ((global_n & 3) == 0 && (reinterpret_cast<uintptr_t>(&C[c_index]) & 15) == 0) {
                    float4 out;
                    out.x = alpha * results[m + wm * TM][0 + wn * TN] + (beta == 0.0f ? 0.0f : beta * C[c_index + 0]);
                    out.y = alpha * results[m + wm * TM][1 + wn * TN] + (beta == 0.0f ? 0.0f : beta * C[c_index + 1]);
                    out.z = alpha * results[m + wm * TM][2 + wn * TN] + (beta == 0.0f ? 0.0f : beta * C[c_index + 2]);
                    out.w = alpha * results[m + wm * TM][3 + wn * TN] + (beta == 0.0f ? 0.0f : beta * C[c_index + 3]);
                    *reinterpret_cast<float4*>(&C[c_index]) = out;
                } else {
                    #pragma unroll
                    for (int n = 0; n < TN; ++n) {
                        const int gn = global_n + n;
                        if (gn < N) {
                            const int ci = OFFSET(global_m, gn, N);
                            C[ci] = alpha * results[m + wm * TM][n + wn * TN] + (beta == 0.0f ? 0.0f : beta * C[ci]);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    const int gn = global_n + n;
                    if (global_m < M && gn < N) {
                        const int ci = OFFSET(global_m, gn, N);
                        C[ci] = alpha * results[m + wm * TM][n + wn * TN] + (beta == 0.0f ? 0.0f : beta * C[ci]);
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
