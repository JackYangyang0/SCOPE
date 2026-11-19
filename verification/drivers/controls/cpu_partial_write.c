/* NEGATIVE control: computes correctly only for the first half of the rows and
 * leaves the rest of C untouched (still holding C0).
 *
 * Trigger case: uniform_small.s00 with pattern C0 (beta = 0 must overwrite all
 * of C).  Must NOT pass.
 */
#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    const int rows = M / 2;
    for (int m = 0; m < rows; ++m) {
        for (int n = 0; n < N; ++n) {
            double accumulator = 0.0;
            for (int k = 0; k < K; ++k) {
                accumulator += (double)A[OFFSET(m, k, K)] * (double)B[OFFSET(k, n, N)];
            }
            const int index = OFFSET(m, n, N);
            C[index] = (float)((double)alpha * accumulator + (double)beta * (double)C[index]);
        }
    }
}