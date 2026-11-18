/* POSITIVE control: FP64 accumulation, one rounding to float32 on store.
 *
 * Expected: passes the main policy on every case, including the coefficient
 * cases.  If this control fails, the evaluator, the driver or the reference is
 * broken — not the kernels under test.
 */
#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    for (int m = 0; m < M; ++m) {
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
