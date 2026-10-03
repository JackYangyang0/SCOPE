#include "kernel.h"
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    /* SCOPE_CPU_PATCH_KERNEL_BEGIN */
    const float * restrict A_data = A;
    const float * restrict B_data = B;
    float * restrict C_data = C;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    const int L2_BLOCK_M = 64;
    const int L2_BLOCK_N = 32;
    const int L2_BLOCK_K = 512;
    const int MR = 4;
    const int NR = 8;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n0 = 0; n0 < N; n0 += L2_BLOCK_N) {
        for (int m0 = 0; m0 < M; m0 += L2_BLOCK_M) {
            const int n_end = (n0 + L2_BLOCK_N < N) ? (n0 + L2_BLOCK_N) : N;
            const int m_end = (m0 + L2_BLOCK_M < M) ? (m0 + L2_BLOCK_M) : M;
            for (int m = m0; m < m_end; m += MR) {
                const int full_m = (m + MR <= m_end);
                for (int n = n0; n < n_end; n += NR) {
                    const int full_n = (n + NR <= n_end);
                    if (full_m && full_n) {
                        __m256 acc00 = _mm256_setzero_ps();
                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc20 = _mm256_setzero_ps();
                        __m256 acc30 = _mm256_setzero_ps();
                        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {
                            const int k_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;
                            for (int k = k0; k < k_end; ++k) {
                                __m256 b0 = _mm256_loadu_ps(&B_data[k * ldb + n + 0]);
                                __m256 a0 = _mm256_broadcast_ss(&A_data[(m + 0) * lda + k]);
                                acc00 = _mm256_fmadd_ps(a0, b0, acc00);
                                __m256 a1 = _mm256_broadcast_ss(&A_data[(m + 1) * lda + k]);
                                acc10 = _mm256_fmadd_ps(a1, b0, acc10);
                                __m256 a2 = _mm256_broadcast_ss(&A_data[(m + 2) * lda + k]);
                                acc20 = _mm256_fmadd_ps(a2, b0, acc20);
                                __m256 a3 = _mm256_broadcast_ss(&A_data[(m + 3) * lda + k]);
                                acc30 = _mm256_fmadd_ps(a3, b0, acc30);
                            }
                        }
                        const __m256 alpha_v = _mm256_set1_ps(alpha);
                        if (beta == 0.0f) {
                            _mm256_storeu_ps(&C_data[(m + 0) * ldc + n + 0], _mm256_mul_ps(alpha_v, acc00));
                            _mm256_storeu_ps(&C_data[(m + 1) * ldc + n + 0], _mm256_mul_ps(alpha_v, acc10));
                            _mm256_storeu_ps(&C_data[(m + 2) * ldc + n + 0], _mm256_mul_ps(alpha_v, acc20));
                            _mm256_storeu_ps(&C_data[(m + 3) * ldc + n + 0], _mm256_mul_ps(alpha_v, acc30));
                        } else {
                            const __m256 beta_v = _mm256_set1_ps(beta);
                            __m256 c00 = _mm256_loadu_ps(&C_data[(m + 0) * ldc + n + 0]);
                            _mm256_storeu_ps(&C_data[(m + 0) * ldc + n + 0], _mm256_fmadd_ps(beta_v, c00, _mm256_mul_ps(alpha_v, acc00)));
                            __m256 c10 = _mm256_loadu_ps(&C_data[(m + 1) * ldc + n + 0]);
                            _mm256_storeu_ps(&C_data[(m + 1) * ldc + n + 0], _mm256_fmadd_ps(beta_v, c10, _mm256_mul_ps(alpha_v, acc10)));
                            __m256 c20 = _mm256_loadu_ps(&C_data[(m + 2) * ldc + n + 0]);
                            _mm256_storeu_ps(&C_data[(m + 2) * ldc + n + 0], _mm256_fmadd_ps(beta_v, c20, _mm256_mul_ps(alpha_v, acc20)));
                            __m256 c30 = _mm256_loadu_ps(&C_data[(m + 3) * ldc + n + 0]);
                            _mm256_storeu_ps(&C_data[(m + 3) * ldc + n + 0], _mm256_fmadd_ps(beta_v, c30, _mm256_mul_ps(alpha_v, acc30)));
                        }
                    } else {
                        const int rm_count = (m + MR <= m_end) ? MR : (m_end - m);
                        const int rn_count = (n + NR <= n_end) ? NR : (n_end - n);
                        float acc[8][32] = {0.0f};
                        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {
                            const int k_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;
                            for (int k = k0; k < k_end; ++k) {
                                for (int rm = 0; rm < rm_count; ++rm) {
                                    const float a_val = A_data[(m + rm) * lda + k];
                                    for (int rn = 0; rn < rn_count; ++rn) {
                                        acc[rm][rn] += a_val * B_data[k * ldb + n + rn];
                                    }
                                }
                            }
                        }
                        for (int rm = 0; rm < rm_count; ++rm) {
                            for (int rn = 0; rn < rn_count; ++rn) {
                                const int c_idx = (m + rm) * ldc + n + rn;
                                C_data[c_idx] = alpha * acc[rm][rn] + beta * C_data[c_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    /* SCOPE_CPU_PATCH_KERNEL_END */
}
