#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>

// SCOPE_CONFIG: generated compile-time parameters precede this file.
namespace scope_specialization {
constexpr int BM = SCOPE_BM, BN = SCOPE_BN, BK = SCOPE_BK;
constexpr int WM = SCOPE_WM, WN = SCOPE_WN, TM = SCOPE_TM, TN = SCOPE_TN;
constexpr int WI = SCOPE_WMITER, NI = SCOPE_WNITER;
constexpr int RM = WM / WI, RN = WN / NI;
constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
constexpr int APITCH = (SCOPE_TRANSPOSE_A ? BM : BK) + SCOPE_PADDING;
constexpr int BPITCH = BN + SCOPE_PADDING;
constexpr int ASIZE = (SCOPE_TRANSPOSE_A ? BK : BM) * APITCH;
constexpr int BSIZE = BK * BPITCH;
static_assert(BM % WM == 0 && BN % WN == 0, "block/warp mismatch");
static_assert(WM % WI == 0 && WN % NI == 0, "warp iteration mismatch");
static_assert(WI % TM == 0 && NI % TN == 0 && (WI/TM)*(NI/TN) == 32, "lane coverage");
static_assert(THREADS <= 1024, "thread limit");

__device__ __forceinline__ int ai(int m, int k) {
    return SCOPE_TRANSPOSE_A ? k * APITCH + m : m * APITCH + k;
}

__device__ __forceinline__ void load_tile(int tid, int m0, int n0, int base,
    int M, int N, int K, const float* A, const float* B, float* sa, float* sb) {
    for (int i=tid; i<BM*BK; i+=THREADS) {
        const int m=i/BK, k=i%BK;
        float* dst = &sa[ai(m,k)];
        if (m0+m < M && base+k < K) {
            const float* src = &A[static_cast<size_t>(m0+m)*K+base+k];
            if constexpr (SCOPE_ASYNC) __pipeline_memcpy_async(dst,src,sizeof(float));
            else *dst = *src;
        } else *dst = 0.0f;
    }
    for (int i=tid; i<BK*BN; i+=THREADS) {
        const int k=i/BN, n=i%BN;
        float* dst = &sb[k*BPITCH+n];
        if (base+k < K && n0+n < N) {
            const float* src = &B[static_cast<size_t>(base+k)*N+n0+n];
            if constexpr (SCOPE_ASYNC) __pipeline_memcpy_async(dst,src,sizeof(float));
            else *dst = *src;
        } else *dst = 0.0f;
    }
    if constexpr (SCOPE_ASYNC) __pipeline_commit();
}

__global__ void kernel(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C, float* partial) {
    __shared__ float sa[SCOPE_STAGES][ASIZE];
    __shared__ float sb[SCOPE_STAGES][BSIZE];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = (warp / (BN/WN)) * WM, wn = (warp % (BN/WN)) * WN;
    const int lm = (lane / (NI/TN)) * TM, ln = (lane % (NI/TN)) * TN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[RM][RN][TM][TN] = {};
    const int kt = (K + BK - 1) / BK;
    const int first = (static_cast<long long>(kt) * blockIdx.z / SCOPE_SPLIT_K) * BK;
    const int last = (static_cast<long long>(kt) * (blockIdx.z+1) / SCOPE_SPLIT_K) * BK;
    if (first < last) load_tile(tid,m0,n0,first,M,N,K,A,B,sa[0],sb[0]);
    if constexpr (SCOPE_ASYNC) __pipeline_wait_prior(0);
    __syncthreads();
    int stage = 0;
    for (int base = first; base < last; base += BK) {
        if constexpr (SCOPE_STAGES == 2)
            if (base+BK < last) load_tile(tid,m0,n0,base+BK,M,N,K,A,B,sa[stage^1],sb[stage^1]);
        #pragma unroll SCOPE_UNROLL
        for (int k = 0; k < BK; ++k) {
            float ar[RM][TM], br[RN][TN];
            #pragma unroll
            for (int r=0; r<RM; ++r)
                #pragma unroll
                for (int m=0; m<TM; ++m) ar[r][m] = sa[stage][ai(wm+r*WI+lm+m,k)];
            #pragma unroll
            for (int r=0; r<RN; ++r)
                #pragma unroll
                for (int n=0; n<TN; ++n) br[r][n] = sb[stage][k*BPITCH+wn+r*NI+ln+n];
            #pragma unroll
            for (int r=0; r<RM; ++r)
                #pragma unroll
                for (int s=0; s<RN; ++s)
                    #pragma unroll
                    for (int m=0; m<TM; ++m)
                        #pragma unroll
                        for (int n=0; n<TN; ++n) acc[r][s][m][n] = fmaf(ar[r][m],br[s][n],acc[r][s][m][n]);
        }
        if constexpr (SCOPE_ASYNC) __pipeline_wait_prior(0);
        __syncthreads();
        if constexpr (SCOPE_STAGES == 2) stage ^= 1;
        else {
            if (base+BK < last) load_tile(tid,m0,n0,base+BK,M,N,K,A,B,sa[0],sb[0]);
            if constexpr (SCOPE_ASYNC) __pipeline_wait_prior(0);
            __syncthreads();
        }
    }
    #pragma unroll
    for (int r=0; r<RM; ++r)
        #pragma unroll
        for (int s=0; s<RN; ++s)
            #pragma unroll
            for (int m=0; m<TM; ++m)
                #pragma unroll
                for (int n=0; n<TN; ++n) {
                    const int row = m0+wm+r*WI+lm+m, col = n0+wn+s*NI+ln+n;
                    if (row < M && col < N) {
                        const size_t pos = static_cast<size_t>(row)*N+col;
                        if constexpr (SCOPE_SPLIT_K > 1)
                            partial[static_cast<size_t>(blockIdx.z)*M*N+pos] = acc[r][s][m][n];
                        else C[pos] = alpha*acc[r][s][m][n] + (beta == 0.0f ? 0.0f : beta*C[pos]);
                    }
                }
}

__global__ void reduce(int M, int N, float alpha, float beta, const float* partial, float* C) {
    const size_t pos = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t size = static_cast<size_t>(M)*N;
    if (pos >= size) return;
    float sum = 0;
    #pragma unroll
    for (int s=0; s<SCOPE_SPLIT_K; ++s) sum += partial[s*size+pos];
    C[pos] = alpha*sum + (beta == 0.0f ? 0.0f : beta*C[pos]);
}

inline void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
}

inline void cuda_gemm(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    using namespace scope_specialization;
    float* partial = nullptr;
    if constexpr (SCOPE_SPLIT_K > 1)
        check(cudaMalloc(reinterpret_cast<void**>(&partial), sizeof(float)*static_cast<size_t>(M)*N*SCOPE_SPLIT_K));
    kernel<<<dim3((N+BN-1)/BN, (M+BM-1)/BM, SCOPE_SPLIT_K), THREADS>>>(M,N,K,alpha,A,B,beta,C,partial);
    check(cudaGetLastError());
    if constexpr (SCOPE_SPLIT_K > 1) {
        reduce<<<(static_cast<size_t>(M)*N+255)/256,256>>>(M,N,alpha,beta,partial,C);
        check(cudaGetLastError());
        check(cudaFree(partial));
    }
}
