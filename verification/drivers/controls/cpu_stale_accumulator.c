/* NEGATIVE control: the accumulator is not reset between output columns, the
 * deterministic stand-in for a missing synchronisation / stale register.
 *
 * Trigger case: uniform_small.s00 with N > 1.  Must NOT pass.
 */
#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    for (int m = 0; m < M; ++m) {
        double accumulator = 0.0;          /* never reset inside the n loop */
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                accumulator += (double)A[OFFSET(m, k, K)] * (double)B[OFFSET(k, n, N)];
            }
            const int index = OFFSET(m, n, N);
            C[index] = (float)((double)alpha * accumulator + (double)beta * (double)C[index]);
        }
    }
}