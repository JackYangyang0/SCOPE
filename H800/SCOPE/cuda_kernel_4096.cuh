
#include <cuda_runtime.h>
#include <cstdint>

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
    {
        const int load_a_smem_m = (tid * 4) / BK;
        const int load_a_smem_k = (tid * 4) % BK;
        const int global_k_base = 0;
        const int global_m = tile_m0 + load_a_smem_m;
        float4 tmp = FLOAT4(A[global_m * K + global_k_base + load_a_smem_k]);
        As[0][load_a_smem_k + 0][load_a_smem_m] = tmp.x;
        As[0][load_a_smem_k + 1][load_a_smem_m] = tmp.y;
        As[0][load_a_smem_k + 2][load_a_smem_m] = tmp.z;
        As[0][load_a_smem_k + 3][load_a_smem_m] = tmp.w;
    }
    for (int iter = 0; iter < 2; ++iter) {
        const int v = tid * 2 + iter;
        const int bk = v / 32;
        const int bn = (v % 32) * 4;
        const int global_k = 0 + bk;
        const int global_n_base = tile_n0 + bn;
        float4 tmp = FLOAT4(B[global_k * N + global_n_base]);
        Bs[0][bk][bn + 0] = tmp.x;
        Bs[0][bk][bn + 1] = tmp.y;
        Bs[0][bk][bn + 2] = tmp.z;
        Bs[0][bk][bn + 3] = tmp.w;
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
        {
            const int load_a_smem_m = (tid * 4) / BK;
            const int load_a_smem_k = (tid * 4) % BK;
            const int global_k_base = bkIdx * BK;
            const int global_m = tile_m0 + load_a_smem_m;
            float4 tmp = FLOAT4(A[global_m * K + global_k_base + load_a_smem_k]);
            As[mem_flag][load_a_smem_k + 0][load_a_smem_m] = tmp.x;
            As[mem_flag][load_a_smem_k + 1][load_a_smem_m] = tmp.y;
            As[mem_flag][load_a_smem_k + 2][load_a_smem_m] = tmp.z;
            As[mem_flag][load_a_smem_k + 3][load_a_smem_m] = tmp.w;
        }
        for (int iter = 0; iter < 2; ++iter) {
            const int v = tid * 2 + iter;
            const int bk = v / 32;
            const int bn = (v % 32) * 4;
            const int global_k = bkIdx * BK + bk;
            const int global_n_base = tile_n0 + bn;
            float4 tmp = FLOAT4(B[global_k * N + global_n_base]);
            Bs[mem_flag][bk][bn + 0] = tmp.x;
            Bs[mem_flag][bk][bn + 1] = tmp.y;
            Bs[mem_flag][bk][bn + 2] = tmp.z;
            Bs[mem_flag][bk][bn + 3] = tmp.w;
        }
    }
    __syncthreads();
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
                const int global_n_base = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN;
                if (global_m < M && global_n_base + 3 < N) {
                    const int c_index = OFFSET(global_m, global_n_base, N);
                    if (reinterpret_cast<std::uintptr_t>(&C[c_index]) % 16 == 0) {
                        float4 out;
                        out.x = alpha * results[m + wm * TM][0 + wn * TN];
                        out.y = alpha * results[m + wm * TM][1 + wn * TN];
                        out.z = alpha * results[m + wm * TM][2 + wn * TN];
                        out.w = alpha * results[m + wm * TM][3 + wn * TN];
                        *reinterpret_cast<float4*>(&C[c_index]) = out;
                    } else {
                        C[c_index + 0] = alpha * results[m + wm * TM][0 + wn * TN];
                        C[c_index + 1] = alpha * results[m + wm * TM][1 + wn * TN];
                        C[c_index + 2] = alpha * results[m + wm * TM][2 + wn * TN];
                        C[c_index + 3] = alpha * results[m + wm * TM][3 + wn * TN];
                    }
                } else if (global_m < M) {
                    #pragma unroll
                    for (int n = 0; n < TN; ++n) {
                        const int global_n = global_n_base + n;
                        if (global_n < N) {
                            const int c_index = OFFSET(global_m, global_n, N);
                            C[c_index] = alpha * results[m + wm * TM][n + wn * TN];
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
    // Ptxas register tuning: 112 registers/thread, maxrregcount=128, estimated occupancy=0.5
    // Based on resource_feedback from ptxas_and_runtime verification
    // Dispatch validation: ensure dimensions divisible by tiles and pointers aligned for float4 loads/stores
    if ((M % BM != 0) || (N % BN != 0) || (K % BK != 0)) {
        return;
    }
    if ((reinterpret_cast<std::uintptr_t>(A) % 16 != 0) ||
        (reinterpret_cast<std::uintptr_t>(B) % 16 != 0) ||
        (reinterpret_cast<std::uintptr_t>(C) % 16 != 0)) {
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
