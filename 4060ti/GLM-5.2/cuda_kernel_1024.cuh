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
    __shared__ __align__(16) float Bs[2][BK][BN];
    static_assert(BM == 128 && BN == 64 && BK == 16, "SCOPE launch config and shared-memory IR disagree");
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
    const int linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int threads = blockDim.x * blockDim.y * blockDim.z;
    const int a_vecs = BM * (BK / 4);
    const int b_vecs = BK * (BN / 4);
    (void)global_m_base;
    (void)global_n_base;
    (void)elements_per_tile;
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
{
    const int k_base = 0;
    #pragma unroll
for (int scope_fixed_load_offset = 0; scope_fixed_load_offset < 512; scope_fixed_load_offset += 256) {
    const int v = linear_tid + scope_fixed_load_offset;
        const int am = v / (BK / 4);
        const int ak = (v % (BK / 4)) * 4;
        const int gk = k_base + ak;
        float4 av = FLOAT4(A[(tile_m0 + am) * K + gk]);
        As[0][am][ak + 0] = av.x;
        As[0][am][ak + 1] = av.y;
        As[0][am][ak + 2] = av.z;
        As[0][am][ak + 3] = av.w;
    }
    #pragma unroll
for (int scope_fixed_load_offset = 0; scope_fixed_load_offset < 256; scope_fixed_load_offset += 256) {
    const int v = linear_tid + scope_fixed_load_offset;
        const int bk = v / (BN / 4);
        const int bn = (v % (BN / 4)) * 4;
        const int gk = k_base + bk;
        float4 bv = FLOAT4(B[gk * N + tile_n0 + bn]);
        FLOAT4(Bs[0][bk][bn]) = bv;
    }
}
/*
     * GLOBAL_TO_SHARED_LOAD_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    __syncthreads(); // Publish the initial cooperative tile.

for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
    const int comp_flag = (bkIdx - 1) & 1;
    const int mem_flag = bkIdx & 1;

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int wm = 0; wm < WM / WMITER; ++wm) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[comp_flag][Wrow * WM + wm * WMITER + Trow * TM + i][k];
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

    {
        const int k_base = bkIdx * BK;
        #pragma unroll
for (int scope_fixed_load_offset = 0; scope_fixed_load_offset < 512; scope_fixed_load_offset += 256) {
    const int v = linear_tid + scope_fixed_load_offset;
            const int am = v / (BK / 4);
            const int ak = (v % (BK / 4)) * 4;
            const int gk = k_base + ak;
            float4 av = FLOAT4(A[(tile_m0 + am) * K + gk]);
            As[mem_flag][am][ak + 0] = av.x;
            As[mem_flag][am][ak + 1] = av.y;
            As[mem_flag][am][ak + 2] = av.z;
            As[mem_flag][am][ak + 3] = av.w;
        }
        #pragma unroll
for (int scope_fixed_load_offset = 0; scope_fixed_load_offset < 256; scope_fixed_load_offset += 256) {
    const int v = linear_tid + scope_fixed_load_offset;
            const int bk = v / (BN / 4);
            const int bn = (v % (BN / 4)) * 4;
            const int gk = k_base + bk;
            float4 bv = FLOAT4(B[gk * N + tile_n0 + bn]);
            FLOAT4(Bs[mem_flag][bk][bn]) = bv;
        }
    }

    __syncthreads();
}

{
    const int comp_flag = (k_tiles - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int wm = 0; wm < WM / WMITER; ++wm) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[comp_flag][Wrow * WM + wm * WMITER + Trow * TM + i][k];
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
            if (global_m < M && global_n + TN - 1 < N) {
                const int c_index = OFFSET(global_m, global_n, N);
                float4 cval;
                cval.x = alpha * results[m + wm * TM][0 + wn * TN];
                cval.y = alpha * results[m + wm * TM][1 + wn * TN];
                cval.z = alpha * results[m + wm * TM][2 + wn * TN];
                cval.w = alpha * results[m + wm * TM][3 + wn * TN];
                if (beta == 0.0f) {
                    FLOAT4(C[c_index]) = cval;
                } else {
                    float4 existing = FLOAT4(C[c_index]);
                    cval.x = cval.x + beta * existing.x;
                    cval.y = cval.y + beta * existing.y;
                    cval.z = cval.z + beta * existing.z;
                    cval.w = cval.w + beta * existing.w;
                    FLOAT4(C[c_index]) = cval;
                }
            } else {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    const int gn = global_n + n;
                    if (global_m < M && gn < N) {
                        const int c_index = OFFSET(global_m, gn, N);
                        C[c_index] = alpha * results[m + wm * TM][n + wn * TN] + beta * C[c_index];
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
    static const int BM = 128;
    static const int BN = 64;
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
