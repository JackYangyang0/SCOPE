/* NEGATIVE control: writes zeros and never reads A, B or C.
 *
 * Trigger case: any configuration whose expected output is non-zero
 * (e.g. uniform_small.s00, pattern C0).  Must NOT pass there.
 * It legitimately passes the `zero` configuration, which is why the evaluator
 * needs all 64 checks rather than a single one.
 */
#include "kernel.h"

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    (void)K; (void)alpha; (void)A; (void)B; (void)beta;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * N + n] = 0.0f;
        }
    }
}
