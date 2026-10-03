#include "kernel.h"
#include "kernel_variants.h"

static int maximum_dimension(int M, int N, int K) {
    int value = M > N ? M : N;
    return value > K ? value : K;
}

void cpu_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C) {
    const int extent = maximum_dimension(M, N, K);
    if (extent <= 256) {
        cpu_gemm_256(M, N, K, alpha, A, B, beta, C);
    } else if (extent <= 512) {
        cpu_gemm_512(M, N, K, alpha, A, B, beta, C);
    } else if (extent <= 768) {
        cpu_gemm_768(M, N, K, alpha, A, B, beta, C);
    } else {
        cpu_gemm_1024(M, N, K, alpha, A, B, beta, C);
    }
}
