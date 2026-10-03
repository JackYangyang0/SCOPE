#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C) {
    /* SCOPE_CPU_PATCH_KERNEL_BEGIN */
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float accumulator = 0.0f;
            for (int k = 0; k < K; ++k) {
                accumulator += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
            }
            const int c_index = OFFSET(m, n, N);
            C[c_index] = alpha * accumulator + beta * C[c_index];
        }
    }
    /* SCOPE_CPU_PATCH_KERNEL_END */
}
