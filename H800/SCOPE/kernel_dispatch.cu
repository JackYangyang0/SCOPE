#include "kernel.h"
#include "kernel_variants.h"

#define gemm scope_gemm_512
#define cuda_gemm cuda_gemm_512
#include "cuda_kernel_512.cuh"
#undef cuda_gemm
#undef gemm

#define gemm scope_gemm_1024
#define cuda_gemm cuda_gemm_1024
#include "cuda_kernel_1024.cuh"
#undef cuda_gemm
#undef gemm

#define gemm scope_gemm_2048
#define cuda_gemm cuda_gemm_2048
#include "cuda_kernel_2048.cuh"
#undef cuda_gemm
#undef gemm

#define gemm scope_gemm_4096
#define cuda_gemm cuda_gemm_4096
#include "cuda_kernel_4096.cuh"
#undef cuda_gemm
#undef gemm

void cuda_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {
    if (M >= 4096 || N >= 4096) {
        cuda_gemm_4096(M, N, K, alpha, A, B, beta, C);
    } else if (M >= 2048 || N >= 2048) {
        cuda_gemm_2048(M, N, K, alpha, A, B, beta, C);
    } else if (M >= 1024 || N >= 1024) {
        cuda_gemm_1024(M, N, K, alpha, A, B, beta, C);
    } else {
        cuda_gemm_512(M, N, K, alpha, A, B, beta, C);
    }
}
