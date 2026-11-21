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
__global__ __launch_bounds__(256, 1) void gemm(
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
__shared__ float As[2][BK][BM];  // K-major, M-contiguous
__shared__ float Bs[2][BK][BN];  // K-major, N-contiguous
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
    const int load_a_smem_m = tid / (BK / 4);
    const int load_a_smem_k = (tid % (BK / 4)) * 4;
    const int load_b_smem_k = tid / (BN / 4);
    const int load_b_smem_n = (tid % (BN / 4)) * 4;
    const int hightA = thread_num / (BK / 4);
    const int hightB = thread_num / (BN / 4);
    (void)global_m_base;
    (void)global_n_base;
    (void)load_a_smem_m;
    (void)load_a_smem_k;
    (void)load_b_smem_n;
    (void)load_b_smem_k;
    (void)hightA;
    (void)hightB;
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
    for (int loadOffset = 0; loadOffset < BM; loadOffset += hightA) {
        if (BM % hightA != 0 && loadOffset + load_a_smem_m >= BM) break;
        float4 tmp = FLOAT4(A[OFFSET(load_a_smem_m + loadOffset, load_a_smem_k, K) + A_offset]);
        As[0][load_a_smem_k][load_a_smem_m + loadOffset] = tmp.x;
        As[0][load_a_smem_k + 1][load_a_smem_m + loadOffset] = tmp.y;
        As[0][load_a_smem_k + 2][load_a_smem_m + loadOffset] = tmp.z;
        As[0][load_a_smem_k + 3][load_a_smem_m + loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += hightB) {
        if (BK % hightB != 0 && loadOffset + load_b_smem_k >= BK) break;
        FLOAT4(Bs[0][load_b_smem_k + loadOffset][load_b_smem_n]) =
            FLOAT4(B[OFFSET(load_b_smem_k + loadOffset, load_b_smem_n, N) + B_offset]);
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
    for (int loadOffset = 0; loadOffset < BM; loadOffset += hightA) {
        if (BM % hightA != 0 && loadOffset + load_a_smem_m >= BM) break;
        float4 tmp = FLOAT4(A[OFFSET(load_a_smem_m + loadOffset, load_a_smem_k + bkIdx * BK, K) + A_offset]);
        As[mem_flag][load_a_smem_k][load_a_smem_m + loadOffset] = tmp.x;
        As[mem_flag][load_a_smem_k + 1][load_a_smem_m + loadOffset] = tmp.y;
        As[mem_flag][load_a_smem_k + 2][load_a_smem_m + loadOffset] = tmp.z;
        As[mem_flag][load_a_smem_k + 3][load_a_smem_m + loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += hightB) {
        if (BK % hightB != 0 && loadOffset + load_b_smem_k >= BK) break;
        FLOAT4(Bs[mem_flag][load_b_smem_k + loadOffset][load_b_smem_n]) =
            FLOAT4(B[OFFSET(load_b_smem_k + loadOffset + bkIdx * BK, load_b_smem_n, N) + B_offset]);
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
    static const int BK = 16;
    static const int WM = 16;
    static const int WN = 64;
    static const int WMITER = 16;
    static const int WNITER = 64;
    static const int TM = 8;
    static const int TN = 4;
    dim3 block((BM * BN) / (WM * WN) * 32, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    /*
     * LAUNCH_CONFIG_END
     */

    dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    // SCOPE_SHARED_LAUNCH_BEGIN
{
  cudaError_t scope_attr_0 = cudaFuncSetAttribute(gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (scope_attr_0 != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(scope_attr_0));
    return;
  }
}
// SCOPE_SHARED_LAUNCH_END
gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);
}
