/* NEGATIVE control: drops the last k term.
 *
 * Trigger case: uniform_small.s00.  Must NOT pass.
 */
#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    const int limit = K > 1 ? K - 1 : K;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double accumulator = 0.0;
            for (int k = 0; k < limit; ++k) {
                accumulator += (double)A[OFFSET(m, k, K)] * (double)B[OFFSET(k, n, N)];
            }
            const int index = OFFSET(m, n, N);
            C[index] = (float)((double)alpha * accumulator + (double)beta * (double)C[index]);
        }
    }
}