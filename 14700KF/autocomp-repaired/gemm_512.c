#include "repaired_blocked_gemm.h"

void gemm_cpu(int M, int N, int K, const float *A, const float *B, float *C) {
    repaired_blocked_gemm(M, N, K, A, B, C, 64, 256, 256, 8, 8);
}
