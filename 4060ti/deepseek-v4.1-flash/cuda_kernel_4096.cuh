#include <stdio.h>
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
extern __shared__ __align__(16) unsigned char scope_shared[];
float (&As)[2][BK][BM + 1] = *reinterpret_cast<float (*)[2][BK][BM + 1]>(scope_shared);
float (&Bs)[2][BK][BN + 1] = *reinterpret_cast<float (*)[2][BK][BN + 1]>(scope_shared + 2 * BK * (BM + 1) * sizeof(float));
static_assert(BM == 64 && BN == 128 && BK == 32, "SCOPE launch config and shared-memory IR disagree");
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
const int threads = blockDim.x * blockDim.y * blockDim.z;
const int linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
const int a_vec_per_row = BK / 4;
const int a_vec_count = BM * a_vec_per_row;
const int b_vec_per_row = BN / 4;
const int b_vec_count = BK * b_vec_per_row;
(void)elements_per_tile;
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
    float regA[TM];
    float regB[TN];
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
for (int v = linear_tid; v < a_vec_count; v += threads) {
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
for (int v = linear_tid; v < b_vec_count; v += threads) {
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
                    regA[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    regB[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        results[wm * TM + i][wn * TN + j] += regA[i] * regB[j];
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
    const int k_base = bkIdx * BK;
    for (int v = linear_tid; v < a_vec_count; v += threads) {
        const int am = v / a_vec_per_row;
        const int ak = (v % a_vec_per_row) * 4;
        const int gm = tile_m0 + am;
        const int gk = k_base + ak;
        const float4 av = FLOAT4(A[OFFSET(gm, gk, K)]);
        As[mem_flag][ak + 0][am] = av.x;
        As[mem_flag][ak + 1][am] = av.y;
        As[mem_flag][ak + 2][am] = av.z;
        As[mem_flag][ak + 3][am] = av.w;
    }
    for (int v = linear_tid; v < b_vec_count; v += threads) {
        const int bk = v / b_vec_per_row;
        const int bn = (v % b_vec_per_row) * 4;
        const int gk = k_base + bk;
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
                regA[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    results[wm * TM + i][wn * TN + j] += regA[i] * regB[j];
                }
            }
        }
    }
}
__syncthreads();
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
    static const int BM = 64;
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
    const bool divisible = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);
    const bool aligned = (reinterpret_cast<uintptr_t>(A) % 16u == 0u) &&
                         (reinterpret_cast<uintptr_t>(B) % 16u == 0u) &&
                         (reinterpret_cast<uintptr_t>(C) % 16u == 0u);
    if (!(divisible && aligned)) {
        return;
    }
/*
     * LAUNCH_CONFIG_END
     */

    dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    // SCOPE_SHARED_LAUNCH_BEGIN
{
  cudaError_t scope_attr_0 = cudaFuncSetAttribute(gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
  if (scope_attr_0 != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(scope_attr_0));
    return;
  }
}
// SCOPE_SHARED_LAUNCH_END
gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN><<<blocksPerGrid, threadsPerBlock,49152>>>(M, N, K, alpha, A, B, beta, C);
}
