/* NEGATIVE control: reads A as if it were column-major (ld = M).
 *
 * Trigger case: uniform_small.s00 (A is not symmetric, so A^T @ B differs).
 * Must NOT pass.  This is the layout half of the §5.4 control pair; the
 * parameter-order half is exercised by the driver's --legacy-mkn-order flag.
 */
#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double accumulator = 0.0;
            for (int k = 0; k < K; ++k) {
                /* wrong: column-major interpretation of A */
                accumulator += (double)A[OFFSET(k, m, M)] * (double)B[OFFSET(k, n, N)];
            }
            const int index = OFFSET(m, n, N);
            C[index] = (float)((double)alpha * accumulator + (double)beta * (double)C[index]);
        }
    }
}