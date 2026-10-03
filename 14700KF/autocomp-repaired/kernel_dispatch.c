#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include "kernel.h"
#include "kernel_variants.h"

typedef void (*raw_gemm_fn)(int, int, int, const float *, const float *, float *);

static raw_gemm_fn select_kernel(int M, int N, int K) {
    int extent = M > N ? M : N;
    if (K > extent) extent = K;
    if (extent <= 256) return gemm_cpu_256_raw;
    if (extent <= 512) return gemm_cpu_512_raw;
    if (extent <= 768) return gemm_cpu_768_raw;
    return gemm_cpu_1024_raw;
}

void cpu_gemm(int M, int N, int K, float alpha,
              const float *A, const float *B, float beta, float *C) {
    raw_gemm_fn kernel = select_kernel(M, N, K);
    if (alpha == 1.0f && beta == 0.0f) {
        kernel(M, N, K, A, B, C);
        return;
    }
    const size_t count = (size_t)M * (size_t)N;
    float *product = (float *)malloc(count * sizeof(float));
    if (!product) return;
    kernel(M, N, K, A, B, product);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) C[i] = alpha * product[i] + beta * C[i];
    free(product);
}
