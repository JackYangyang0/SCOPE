#include "kernel.h"
#include <immintrin.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

static void scope_cpu_scalar_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
            C[OFFSET(m, n, N)] = alpha * sum + beta * C[OFFSET(m, n, N)];
        }
}

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    /* SCOPE_CPU_PATCH_KERNEL_BEGIN */
    const int L2_BLOCK_M = 32;
    const int L2_BLOCK_N = 96;
    const int L2_BLOCK_K = 64;
    const int MR = 4;
    const int NR = 16;
    const int ROW_BLOCK_M = (L2_BLOCK_M < MR * 8) ? L2_BLOCK_M : MR * 8;
    const int full_n = (N / NR) * NR;
    const size_t packed_b_count = (size_t)(full_n / NR) * (size_t)K * (size_t)NR;
    float *packed_b = (float *)_mm_malloc(packed_b_count * sizeof(float), 64);
    if (!packed_b) { scope_cpu_scalar_gemm(M, N, K, alpha, A, B, beta, C); return; }

    #pragma omp parallel for schedule(static)
    for (int nb = 0; nb < full_n / NR; ++nb) {
        const int n = nb * NR;
        float *b_panel = packed_b + (size_t)nb * (size_t)K * (size_t)NR;
        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {
            const int k_end = (k0 + L2_BLOCK_K < K) ? k0 + L2_BLOCK_K : K;
            for (int k = k0; k < k_end; ++k)
                for (int j = 0; j < NR; ++j) b_panel[(size_t)k * NR + j] = B[OFFSET(k, n + j, N)];
        }
    }

    /* SCOPE_CPU_READONLY_PACKED_B: packed_b is immutable below. */
    #pragma omp parallel
    {
        float *a_panel = (float *)_mm_malloc((size_t)MR * (size_t)K * sizeof(float), 64);
        #pragma omp for schedule(static)
        for (int m0 = 0; m0 < M; m0 += ROW_BLOCK_M) {
            const int m_end = (m0 + ROW_BLOCK_M < M) ? m0 + ROW_BLOCK_M : M;
            for (int m = m0; m < m_end; m += MR) {
                const int rm_count = (m + MR <= M) ? MR : M - m;
                if (a_panel && rm_count == MR) {
                    for (int r = 0; r < MR; ++r)
                        for (int k = 0; k < K; ++k) a_panel[(size_t)r * K + k] = A[OFFSET(m + r, k, K)];
                }
                for (int n0 = 0; n0 < full_n; n0 += L2_BLOCK_N) {
                    const int n_end = (n0 + L2_BLOCK_N < full_n) ? n0 + L2_BLOCK_N : full_n;
                    for (int n = n0; n < n_end; n += NR) {
                        if (rm_count == MR) {
                            __m256 acc00 = _mm256_setzero_ps();
                            __m256 acc01 = _mm256_setzero_ps();
                            __m256 acc10 = _mm256_setzero_ps();
                            __m256 acc11 = _mm256_setzero_ps();
                            __m256 acc20 = _mm256_setzero_ps();
                            __m256 acc21 = _mm256_setzero_ps();
                            __m256 acc30 = _mm256_setzero_ps();
                            __m256 acc31 = _mm256_setzero_ps();
                            const float *b_panel = packed_b + (size_t)(n / NR) * (size_t)K * (size_t)NR;
                            for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {
                                const int k_end = (k0 + L2_BLOCK_K < K) ? k0 + L2_BLOCK_K : K;
                                int k = k0;
                                for (; k + 7 < k_end; k += 8) {
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 0) * NR + 32], _MM_HINT_T0);
                                    __m256 b0_0 = _mm256_load_ps(&b_panel[(size_t)(k + 0) * NR + 0]);
                                    __m256 b0_1 = _mm256_load_ps(&b_panel[(size_t)(k + 0) * NR + 8]);
                                    __m256 a0_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 0] : &A[OFFSET(m + 0, k + 0, K)]);
                                    acc00 = _mm256_fmadd_ps(a0_0, b0_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a0_0, b0_1, acc01);
                                    __m256 a0_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 0] : &A[OFFSET(m + 1, k + 0, K)]);
                                    acc10 = _mm256_fmadd_ps(a0_1, b0_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a0_1, b0_1, acc11);
                                    __m256 a0_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 0] : &A[OFFSET(m + 2, k + 0, K)]);
                                    acc20 = _mm256_fmadd_ps(a0_2, b0_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a0_2, b0_1, acc21);
                                    __m256 a0_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 0] : &A[OFFSET(m + 3, k + 0, K)]);
                                    acc30 = _mm256_fmadd_ps(a0_3, b0_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a0_3, b0_1, acc31);
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 1) * NR + 32], _MM_HINT_T0);
                                    __m256 b1_0 = _mm256_load_ps(&b_panel[(size_t)(k + 1) * NR + 0]);
                                    __m256 b1_1 = _mm256_load_ps(&b_panel[(size_t)(k + 1) * NR + 8]);
                                    __m256 a1_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 1] : &A[OFFSET(m + 0, k + 1, K)]);
                                    acc00 = _mm256_fmadd_ps(a1_0, b1_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a1_0, b1_1, acc01);
                                    __m256 a1_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 1] : &A[OFFSET(m + 1, k + 1, K)]);
                                    acc10 = _mm256_fmadd_ps(a1_1, b1_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a1_1, b1_1, acc11);
                                    __m256 a1_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 1] : &A[OFFSET(m + 2, k + 1, K)]);
                                    acc20 = _mm256_fmadd_ps(a1_2, b1_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a1_2, b1_1, acc21);
                                    __m256 a1_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 1] : &A[OFFSET(m + 3, k + 1, K)]);
                                    acc30 = _mm256_fmadd_ps(a1_3, b1_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a1_3, b1_1, acc31);
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 2) * NR + 32], _MM_HINT_T0);
                                    __m256 b2_0 = _mm256_load_ps(&b_panel[(size_t)(k + 2) * NR + 0]);
                                    __m256 b2_1 = _mm256_load_ps(&b_panel[(size_t)(k + 2) * NR + 8]);
                                    __m256 a2_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 2] : &A[OFFSET(m + 0, k + 2, K)]);
                                    acc00 = _mm256_fmadd_ps(a2_0, b2_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a2_0, b2_1, acc01);
                                    __m256 a2_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 2] : &A[OFFSET(m + 1, k + 2, K)]);
                                    acc10 = _mm256_fmadd_ps(a2_1, b2_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a2_1, b2_1, acc11);
                                    __m256 a2_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 2] : &A[OFFSET(m + 2, k + 2, K)]);
                                    acc20 = _mm256_fmadd_ps(a2_2, b2_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a2_2, b2_1, acc21);
                                    __m256 a2_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 2] : &A[OFFSET(m + 3, k + 2, K)]);
                                    acc30 = _mm256_fmadd_ps(a2_3, b2_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a2_3, b2_1, acc31);
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 3) * NR + 32], _MM_HINT_T0);
                                    __m256 b3_0 = _mm256_load_ps(&b_panel[(size_t)(k + 3) * NR + 0]);
                                    __m256 b3_1 = _mm256_load_ps(&b_panel[(size_t)(k + 3) * NR + 8]);
                                    __m256 a3_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 3] : &A[OFFSET(m + 0, k + 3, K)]);
                                    acc00 = _mm256_fmadd_ps(a3_0, b3_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a3_0, b3_1, acc01);
                                    __m256 a3_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 3] : &A[OFFSET(m + 1, k + 3, K)]);
                                    acc10 = _mm256_fmadd_ps(a3_1, b3_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a3_1, b3_1, acc11);
                                    __m256 a3_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 3] : &A[OFFSET(m + 2, k + 3, K)]);
                                    acc20 = _mm256_fmadd_ps(a3_2, b3_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a3_2, b3_1, acc21);
                                    __m256 a3_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 3] : &A[OFFSET(m + 3, k + 3, K)]);
                                    acc30 = _mm256_fmadd_ps(a3_3, b3_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a3_3, b3_1, acc31);
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 4) * NR + 32], _MM_HINT_T0);
                                    __m256 b4_0 = _mm256_load_ps(&b_panel[(size_t)(k + 4) * NR + 0]);
                                    __m256 b4_1 = _mm256_load_ps(&b_panel[(size_t)(k + 4) * NR + 8]);
                                    __m256 a4_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 4] : &A[OFFSET(m + 0, k + 4, K)]);
                                    acc00 = _mm256_fmadd_ps(a4_0, b4_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a4_0, b4_1, acc01);
                                    __m256 a4_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 4] : &A[OFFSET(m + 1, k + 4, K)]);
                                    acc10 = _mm256_fmadd_ps(a4_1, b4_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a4_1, b4_1, acc11);
                                    __m256 a4_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 4] : &A[OFFSET(m + 2, k + 4, K)]);
                                    acc20 = _mm256_fmadd_ps(a4_2, b4_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a4_2, b4_1, acc21);
                                    __m256 a4_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 4] : &A[OFFSET(m + 3, k + 4, K)]);
                                    acc30 = _mm256_fmadd_ps(a4_3, b4_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a4_3, b4_1, acc31);
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 5) * NR + 32], _MM_HINT_T0);
                                    __m256 b5_0 = _mm256_load_ps(&b_panel[(size_t)(k + 5) * NR + 0]);
                                    __m256 b5_1 = _mm256_load_ps(&b_panel[(size_t)(k + 5) * NR + 8]);
                                    __m256 a5_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 5] : &A[OFFSET(m + 0, k + 5, K)]);
                                    acc00 = _mm256_fmadd_ps(a5_0, b5_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a5_0, b5_1, acc01);
                                    __m256 a5_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 5] : &A[OFFSET(m + 1, k + 5, K)]);
                                    acc10 = _mm256_fmadd_ps(a5_1, b5_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a5_1, b5_1, acc11);
                                    __m256 a5_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 5] : &A[OFFSET(m + 2, k + 5, K)]);
                                    acc20 = _mm256_fmadd_ps(a5_2, b5_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a5_2, b5_1, acc21);
                                    __m256 a5_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 5] : &A[OFFSET(m + 3, k + 5, K)]);
                                    acc30 = _mm256_fmadd_ps(a5_3, b5_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a5_3, b5_1, acc31);
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 6) * NR + 32], _MM_HINT_T0);
                                    __m256 b6_0 = _mm256_load_ps(&b_panel[(size_t)(k + 6) * NR + 0]);
                                    __m256 b6_1 = _mm256_load_ps(&b_panel[(size_t)(k + 6) * NR + 8]);
                                    __m256 a6_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 6] : &A[OFFSET(m + 0, k + 6, K)]);
                                    acc00 = _mm256_fmadd_ps(a6_0, b6_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a6_0, b6_1, acc01);
                                    __m256 a6_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 6] : &A[OFFSET(m + 1, k + 6, K)]);
                                    acc10 = _mm256_fmadd_ps(a6_1, b6_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a6_1, b6_1, acc11);
                                    __m256 a6_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 6] : &A[OFFSET(m + 2, k + 6, K)]);
                                    acc20 = _mm256_fmadd_ps(a6_2, b6_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a6_2, b6_1, acc21);
                                    __m256 a6_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 6] : &A[OFFSET(m + 3, k + 6, K)]);
                                    acc30 = _mm256_fmadd_ps(a6_3, b6_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a6_3, b6_1, acc31);
                                    _mm_prefetch((const char *)&b_panel[(size_t)(k + 7) * NR + 32], _MM_HINT_T0);
                                    __m256 b7_0 = _mm256_load_ps(&b_panel[(size_t)(k + 7) * NR + 0]);
                                    __m256 b7_1 = _mm256_load_ps(&b_panel[(size_t)(k + 7) * NR + 8]);
                                    __m256 a7_0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k + 7] : &A[OFFSET(m + 0, k + 7, K)]);
                                    acc00 = _mm256_fmadd_ps(a7_0, b7_0, acc00);
                                    acc01 = _mm256_fmadd_ps(a7_0, b7_1, acc01);
                                    __m256 a7_1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k + 7] : &A[OFFSET(m + 1, k + 7, K)]);
                                    acc10 = _mm256_fmadd_ps(a7_1, b7_0, acc10);
                                    acc11 = _mm256_fmadd_ps(a7_1, b7_1, acc11);
                                    __m256 a7_2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k + 7] : &A[OFFSET(m + 2, k + 7, K)]);
                                    acc20 = _mm256_fmadd_ps(a7_2, b7_0, acc20);
                                    acc21 = _mm256_fmadd_ps(a7_2, b7_1, acc21);
                                    __m256 a7_3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k + 7] : &A[OFFSET(m + 3, k + 7, K)]);
                                    acc30 = _mm256_fmadd_ps(a7_3, b7_0, acc30);
                                    acc31 = _mm256_fmadd_ps(a7_3, b7_1, acc31);
                                }
                                for (; k < k_end; ++k) {
                                    __m256 bt0 = _mm256_load_ps(&b_panel[(size_t)k * NR + 0]);
                                    __m256 bt1 = _mm256_load_ps(&b_panel[(size_t)k * NR + 8]);
                                    __m256 at0 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)0 * K + k] : &A[OFFSET(m + 0, k, K)]);
                                    acc00 = _mm256_fmadd_ps(at0, bt0, acc00);
                                    acc01 = _mm256_fmadd_ps(at0, bt1, acc01);
                                    __m256 at1 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)1 * K + k] : &A[OFFSET(m + 1, k, K)]);
                                    acc10 = _mm256_fmadd_ps(at1, bt0, acc10);
                                    acc11 = _mm256_fmadd_ps(at1, bt1, acc11);
                                    __m256 at2 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)2 * K + k] : &A[OFFSET(m + 2, k, K)]);
                                    acc20 = _mm256_fmadd_ps(at2, bt0, acc20);
                                    acc21 = _mm256_fmadd_ps(at2, bt1, acc21);
                                    __m256 at3 = _mm256_broadcast_ss(a_panel ? &a_panel[(size_t)3 * K + k] : &A[OFFSET(m + 3, k, K)]);
                                    acc30 = _mm256_fmadd_ps(at3, bt0, acc30);
                                    acc31 = _mm256_fmadd_ps(at3, bt1, acc31);
                                }
                            }
                            const __m256 alpha_v = _mm256_set1_ps(alpha);
                            const __m256 beta_v = _mm256_set1_ps(beta);
                            __m256 out00 = _mm256_mul_ps(alpha_v, acc00);
                            if (beta != 0.0f) out00 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 0, n + 0, N)]), out00);
                            _mm256_storeu_ps(&C[OFFSET(m + 0, n + 0, N)], out00);
                            __m256 out01 = _mm256_mul_ps(alpha_v, acc01);
                            if (beta != 0.0f) out01 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 0, n + 8, N)]), out01);
                            _mm256_storeu_ps(&C[OFFSET(m + 0, n + 8, N)], out01);
                            __m256 out10 = _mm256_mul_ps(alpha_v, acc10);
                            if (beta != 0.0f) out10 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 1, n + 0, N)]), out10);
                            _mm256_storeu_ps(&C[OFFSET(m + 1, n + 0, N)], out10);
                            __m256 out11 = _mm256_mul_ps(alpha_v, acc11);
                            if (beta != 0.0f) out11 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 1, n + 8, N)]), out11);
                            _mm256_storeu_ps(&C[OFFSET(m + 1, n + 8, N)], out11);
                            __m256 out20 = _mm256_mul_ps(alpha_v, acc20);
                            if (beta != 0.0f) out20 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 2, n + 0, N)]), out20);
                            _mm256_storeu_ps(&C[OFFSET(m + 2, n + 0, N)], out20);
                            __m256 out21 = _mm256_mul_ps(alpha_v, acc21);
                            if (beta != 0.0f) out21 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 2, n + 8, N)]), out21);
                            _mm256_storeu_ps(&C[OFFSET(m + 2, n + 8, N)], out21);
                            __m256 out30 = _mm256_mul_ps(alpha_v, acc30);
                            if (beta != 0.0f) out30 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 3, n + 0, N)]), out30);
                            _mm256_storeu_ps(&C[OFFSET(m + 3, n + 0, N)], out30);
                            __m256 out31 = _mm256_mul_ps(alpha_v, acc31);
                            if (beta != 0.0f) out31 = _mm256_fmadd_ps(beta_v, _mm256_loadu_ps(&C[OFFSET(m + 3, n + 8, N)]), out31);
                            _mm256_storeu_ps(&C[OFFSET(m + 3, n + 8, N)], out31);
                        } else {
                            for (int r = 0; r < rm_count; ++r)
                                for (int j = 0; j < NR; ++j) {
                                    float sum = 0.0f;
                                    for (int k = 0; k < K; ++k) sum += A[OFFSET(m + r, k, K)] * B[OFFSET(k, n + j, N)];
                                    C[OFFSET(m + r, n + j, N)] = alpha * sum + beta * C[OFFSET(m + r, n + j, N)];
                                }
                        }
                    }
                }
                for (int n = full_n; n < N; ++n)
                    for (int r = 0; r < rm_count; ++r) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) sum += A[OFFSET(m + r, k, K)] * B[OFFSET(k, n, N)];
                        C[OFFSET(m + r, n, N)] = alpha * sum + beta * C[OFFSET(m + r, n, N)];
                    }
            }
        }
        if (a_panel) _mm_free(a_panel);
    }
    _mm_free(packed_b);
    /* SCOPE_CPU_PATCH_KERNEL_END */
}
