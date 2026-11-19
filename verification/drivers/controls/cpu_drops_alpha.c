/* NEGATIVE control: ignores alpha (behaves as alpha = 1).
 *
 * Trigger: the coefficient group with alpha = -0.75.  The main group (alpha = 1)
 * is expected to pass, which is exactly why the coefficient checks exist.
 */
#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    (void)alpha;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double accumulator = 0.0;
            for (int k = 0; k < K; ++k) {
                accumulator += (double)A[OFFSET(m, k, K)] * (double)B[OFFSET(k, n, N)];
            }
            const int index = OFFSET(m, n, N);
            C[index] = (float)(accumulator + (double)beta * (double)C[index]);
        }
    }
}