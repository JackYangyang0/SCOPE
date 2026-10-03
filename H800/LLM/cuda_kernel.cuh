#pragma once

#include <cuda_runtime.h>

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

#ifndef OFFSET
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif

#ifndef FLOAT4
#define FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])
#endif

template <
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int TM,
    int TN>
__global__ void __launch_bounds__(512, 4) gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    
    const int warps_n = BN / WN;
    const int warp_m = warp_id / warps_n;
    const int warp_n = warp_id % warps_n;
    
    const int lane_m = (lane_id / (WN / TN)) * TM;
    const int lane_n = (lane_id % (WN / TN)) * TN;
    
    const int global_m = block_m + warp_m * WM + lane_m;
    const int global_n = block_n + warp_n * WN + lane_n;
    
    float acc[TM][TN] = {{0.0f}};
    
    const int k_tiles = (K + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile & 1;
        const int k_offset = k_tile * BK;
        
        #pragma unroll
        for (int i = tid; i < BM * (BK / 4); i += num_threads) {
            const int m = i / (BK / 4);
            const int k_vec = i % (BK / 4);
            const int gm = block_m + m;
            const int gk = k_offset + k_vec * 4;
            
            if (gm < M && gk + 3 < K && (K & 3) == 0) {
                float4 val = FLOAT4(A[OFFSET(gm, gk, K)]);
                As[stage][m][k_vec * 4 + 0] = val.x;
                As[stage][m][k_vec * 4 + 1] = val.y;
                As[stage][m][k_vec * 4 + 2] = val.z;
                As[stage][m][k_vec * 4 + 3] = val.w;
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    As[stage][m][k_vec * 4 + k] = (gm < M && gk + k < K) ? A[OFFSET(gm, gk + k, K)] : 0.0f;
                }
            }
        }
        
        #pragma unroll
        for (int i = tid; i < BK * (BN / 4); i += num_threads) {
            const int k_row = i / (BN / 4);
            const int n_vec = i % (BN / 4);
            const int gk = k_offset + k_row;
            const int gn = block_n + n_vec * 4;
            
            if (gk < K && gn + 3 < N && (N & 3) == 0) {
                float4 val = FLOAT4(B[OFFSET(gk, gn, N)]);
                Bs[stage][k_row][n_vec * 4 + 0] = val.x;
                Bs[stage][k_row][n_vec * 4 + 1] = val.y;
                Bs[stage][k_row][n_vec * 4 + 2] = val.z;
                Bs[stage][k_row][n_vec * 4 + 3] = val.w;
            } else {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    Bs[stage][k_row][n_vec * 4 + n] = (gk < K && gn + n < N) ? B[OFFSET(gk, gn + n, N)] : 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_vals[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_vals[i] = As[stage][warp_m * WM + lane_m + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                float b_val = Bs[stage][k][warp_n * WN + lane_n + j];
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    acc[i][j] += a_vals[i] * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gm = global_m + i;
            const int gn = global_n + j;
            
            if (gm < M && gn < N) {
                const int idx = OFFSET(gm, gn, N);
                float val = alpha * acc[i][j];
                if (beta != 0.0f) {
                    val += beta * C[idx];
                }
                C[idx] = val;
            }
        }
    }
}

template <
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int TM,
    int TN>
__global__ void __launch_bounds__(256, 8) gemm_kernel_small(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    
    const int warps_n = BN / WN;
    const int warp_m = warp_id / warps_n;
    const int warp_n = warp_id % warps_n;
    
    const int lane_m = (lane_id / (WN / TN)) * TM;
    const int lane_n = (lane_id % (WN / TN)) * TN;
    
    const int global_m = block_m + warp_m * WM + lane_m;
    const int global_n = block_n + warp_n * WN + lane_n;
    
    float acc[TM][TN] = {{0.0f}};
    
    const int k_tiles = (K + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile & 1;
        const int k_offset = k_tile * BK;
        
        #pragma unroll
        for (int i = tid; i < BM * (BK / 4); i += num_threads) {
            const int m = i / (BK / 4);
            const int k_vec = i % (BK / 4);
            const int gm = block_m + m;
            const int gk = k_offset + k_vec * 4;
            
            if (gm < M && gk + 3 < K && (K & 3) == 0) {
                float4 val = FLOAT4(A[OFFSET(gm, gk, K)]);
                As[stage][m][k_vec * 4 + 0] = val.x;
                As[stage][m][k_vec * 4 + 1] = val.y;
                As[stage][m][k_vec * 4 + 2] = val.z;
                As[stage][m][k_vec * 4 + 3] = val.w;
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    As[stage][m][k_vec * 4 + k] = (gm < M && gk + k < K) ? A[OFFSET(gm, gk + k, K)] : 0.0f;
                }
            }
        }
        
        #pragma unroll
        for (int i = tid; i < BK * (BN / 4); i += num_threads) {
            const int k_row = i / (BN / 4);
            const int n_vec = i % (BN / 4);
            const int gk = k_offset + k_row;
            const int gn = block_n + n_vec * 4;
            
            if (gk < K && gn + 3 < N && (N & 3) == 0) {
                float4 val = FLOAT4(B[OFFSET(gk, gn, N)]);
                Bs[stage][k_row][n_vec * 4 + 0] = val.x;
                Bs[stage][k_row][n_vec * 4 + 1] = val.y;
                Bs[stage][k_row][n_vec * 4 + 2] = val.z;
                Bs[stage][k_row][n_vec * 4 + 3] = val.w;
            } else {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    Bs[stage][k_row][n_vec * 4 + n] = (gk < K && gn + n < N) ? B[OFFSET(gk, gn + n, N)] : 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_vals[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_vals[i] = As[stage][warp_m * WM + lane_m + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                float b_val = Bs[stage][k][warp_n * WN + lane_n + j];
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    acc[i][j] += a_vals[i] * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gm = global_m + i;
            const int gn = global_n + j;
            
            if (gm < M && gn < N) {
                const int idx = OFFSET(gm, gn, N);
                float val = alpha * acc[i][j];
                if (beta != 0.0f) {
                    val += beta * C[idx];
                }
                C[idx] = val;
            }
        }
    }
}

