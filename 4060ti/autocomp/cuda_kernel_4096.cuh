#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;
    
    __shared__ half As[BM][BK + 8];
    __shared__ half Bs[BK][BN + 8];
    __shared__ float Cs[BM][BN + 8];
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    
    int num_tiles = (N + BK - 1) / BK;
    
    for (int t = 0; t < num_tiles; t++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 128;
            int r = idx / (BK / 4);
            int c = (idx % (BK / 4)) * 4;
            
            int global_r = by * BM + r;
            int global_c = t * BK + c;
            
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_r < N && global_c + 3 < N) {
                val = *reinterpret_cast<const float4*>(&A[global_r * N + global_c]);
            } else if (global_r < N) {
                if (global_c < N) val.x = A[global_r * N + global_c];
                if (global_c + 1 < N) val.y = A[global_r * N + global_c + 1];
                if (global_c + 2 < N) val.z = A[global_r * N + global_c + 2];
                if (global_c + 3 < N) val.w = A[global_r * N + global_c + 3];
            }
            
            half2 h0 = __floats2half2_rn(val.x, val.y);
            half2 h1 = __floats2half2_rn(val.z, val.w);
            *reinterpret_cast<half2*>(&As[r][c]) = h0;
            *reinterpret_cast<half2*>(&As[r][c + 2]) = h1;
        }
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 128;
            int r = idx / (BN / 4);
            int c = (idx % (BN / 4)) * 4;
            
            int global_r = t * BK + r;
            int global_c = bx * BN + c;
            
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_r < N && global_c + 3 < N) {
                val = *reinterpret_cast<const float4*>(&B[global_r * N + global_c]);
            } else if (global_r < N) {
                if (global_c < N) val.x = B[global_r * N + global_c];
                if (global_c + 1 < N) val.y = B[global_r * N + global_c + 1];
                if (global_c + 2 < N) val.z = B[global_r * N + global_c + 2];
                if (global_c + 3 < N) val.w = B[global_r * N + global_c + 3];
            }
            
            half2 h0 = __floats2half2_rn(val.x, val.y);
            half2 h1 = __floats2half2_rn(val.z, val.w);
            *reinterpret_cast<half2*>(&Bs[r][c]) = h0;
            *reinterpret_cast<half2*>(&Bs[r][c + 2]) = h1;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
            
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(a_frag[i], &As[warp_row * 32 + i * 16][k], BK + 8);
            }
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::load_matrix_sync(b_frag[j], &Bs[k][warp_col * 32 + j * 16], BN + 8);
            }
            
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(&Cs[warp_row * 32 + i * 16][warp_col * 32 + j * 16], c_frag[i][j], BN + 8, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int r = idx / (BN / 4);
        int c = (idx % (BN / 4)) * 4;
        int global_r = by * BM + r;
        int global_c = bx * BN + c;
        
        if (global_r < N && global_c + 3 < N) {
            float4 val = *reinterpret_cast<float4*>(&Cs[r][c]);
            *reinterpret_cast<float4*>(&C[global_r * N + global_c]) = val;
        } else if (global_r < N) {
            if (global_c < N) C[global_r * N + global_c] = Cs[r][c];
            if (global_c + 1 < N) C[global_r * N + global_c + 1] = Cs[r][c + 1];
            if (global_c + 2 < N) C[global_r * N + global_c + 2] = Cs[r][c + 2];
            if (global_c + 3 < N) C[global_r * N + global_c + 3] = Cs[r][c + 3];
        }
    }
}


extern "C" __declspec(dllexport) void launch_gemm(const float* A, const float* B, float* C, int N);
extern "C" __declspec(dllexport) void cuda_sync();

extern "C" __declspec(dllexport) void launch_gemm(const float* A, const float* B, float* C, int N) {
    dim3 block(128);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    gemm_kernel<<<grid, block>>>(A, B, C, N);
}

extern "C" __declspec(dllexport) void cuda_sync() {
    cudaDeviceSynchronize();
}

