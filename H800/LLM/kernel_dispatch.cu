#include "kernel.h"
#include "kernel_variants.h"
#include "cuda_kernel.cuh"
#include "cuda_kernel_512.cuh"
#include "cuda_kernel_1024.cuh"
#include "cuda_kernel_2048.cuh"
#include "cuda_kernel_4096.cuh"

void cuda_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {
    if (M == 512 && N == 512 && K == 512) {
        cuda_gemm_512(M, N, K, alpha, A, B, beta, C);
    } else if (M == 1024 && N == 1024 && K == 1024) {
        cuda_gemm_1024(M, N, K, alpha, A, B, beta, C);
    } else if (M == 2048 && N == 2048 && K == 2048) {
        cuda_gemm_2048(M, N, K, alpha, A, B, beta, C);
    } else if (M == 4096 && N == 4096 && K == 4096) {
        cuda_gemm_4096(M, N, K, alpha, A, B, beta, C);
    } else if (M <= 512 && N <= 512) {
        cuda_gemm_512(M, N, K, alpha, A, B, beta, C);
    } else if (M <= 1024 && N <= 1024) {
        cuda_gemm_1024(M, N, K, alpha, A, B, beta, C);
    } else if (M <= 2048 && N <= 2048) {
        cuda_gemm_2048(M, N, K, alpha, A, B, beta, C);
    } else {
        cuda_gemm_4096(M, N, K, alpha, A, B, beta, C);
    }
}
