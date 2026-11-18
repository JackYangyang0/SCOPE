/* NEGATIVE control: ignores beta and always adds C to the product, i.e. it
 * behaves as if beta were 1.
 *
 * Trigger case: uniform_small.s00 with pattern C0 (beta = 0).  Must NOT pass.
 */
#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    (void)beta;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double accumulator = 0.0;
            for (int k = 0; k < K; ++k) {
                accumulator += (double)A[OFFSET(m, k, K)] * (double)B[OFFSET(k, n, N)];
            }
            const int index = OFFSET(m, n, N);
            C[index] = (float)((double)alpha * accumulator + (double)C[index]);
        }
    }
}