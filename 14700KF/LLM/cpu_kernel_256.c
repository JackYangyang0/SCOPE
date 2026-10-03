#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define MR 8
#define NR 8
#define MC_MAX 256
#define NC_MAX 1024
#define KC_MAX 128

#define ZEROU _mm256_setzero_ps
#define LOADU _mm256_loadu_ps
#define STOREU _mm256_storeu_ps
#define SET1 _mm256_set1_ps
#define FMADD _mm256_fmadd_ps
#define MUL _mm256_mul_ps

static void scalar_gemm(int m, int n, int k, float alpha, const float *a, int lda, 
                        const float *b, int ldb, float beta, float *c, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) sum += a[i * lda + l] * b[l * ldb + j];
            if (beta == 0.0f) c[i * ldc + j] = alpha * sum;
            else c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
        }
    }
}

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            #pragma omp parallel for
            for (int i = 0; i < M * N; i++) C[i] = 0.0f;
        } else if (beta != 1.0f) {
            #pragma omp parallel for
            for (int i = 0; i < M * N; i++) C[i] *= beta;
        }
        return;
    }

    int MC = MC_MAX, NC = 256, KC = KC_MAX;
    if (N >= 512) NC = 512;
    if (N >= 1024) NC = 1024;

    int n_threads = omp_get_max_threads();
    float **packA = (float **)malloc(n_threads * sizeof(float *));
    float **packB = (float **)malloc(n_threads * sizeof(float *));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        packA[tid] = (float *)_mm_malloc(MC_MAX * KC_MAX * sizeof(float), 32);
        packB[tid] = (float *)_mm_malloc(KC_MAX * NC_MAX * sizeof(float), 32);
    }

    int beta_zero = (beta == 0.0f);
    int alpha_one = (alpha == 1.0f);
    __m256 v_alpha = SET1(alpha);
    __m256 v_beta = SET1(beta);

    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < M; i0 += MC) {
        int tid = omp_get_thread_num();
        float *pA = packA[tid];
        float *pB = packB[tid];
        int mc = (i0 + MC > M) ? (M - i0) : MC;

        for (int k0 = 0; k0 < K; k0 += KC) {
            int kc = (k0 + KC > K) ? (K - k0) : KC;

            // Pack A: MC x KC -> pA (KC x MC layout)
            for (int k = 0; k < kc; k++) {
                const float *a_col = A + (i0)*K + (k0 + k);
                float *pA_col = pA + k * MC_MAX;
                for (int i = 0; i < mc; i++) pA_col[i] = a_col[i * K];
            }

            for (int j0 = 0; j0 < N; j0 += NC) {
                int nc = (j0 + NC > N) ? (N - j0) : NC;

                // Pack B: KC x NC -> pB (KC x NC layout)
                for (int k = 0; k < kc; k++) {
                    const float *b_row = B + (k0 + k) * N + j0;
                    float *pB_row = pB + k * NC_MAX;
                    for (int j = 0; j < nc; j++) pB_row[j] = b_row[j];
                }

                // Vectorize across contiguous columns of row-major C and packed B.
                for (int ii = 0; ii < mc; ii++) {
                    int jj = 0;
                    for (; jj + NR <= nc; jj += NR) {
                        __m256 acc = ZEROU();
                        for (int k = 0; k < kc; k++) {
                            __m256 vb = LOADU(pB + k * NC_MAX + jj);
                            acc = FMADD(SET1(pA[k * MC_MAX + ii]), vb, acc);
                        }
                        if (!alpha_one) acc = MUL(acc, v_alpha);
                        float *c_ptr = C + (i0 + ii) * N + (j0 + jj);
                        if (k0 == 0) {
                            if (!beta_zero) acc = FMADD(LOADU(c_ptr), v_beta, acc);
                        } else {
                            acc = _mm256_add_ps(LOADU(c_ptr), acc);
                        }
                        STOREU(c_ptr, acc);
                    }
                    for (; jj < nc; jj++) {
                        float sum = 0.0f;
                        for (int k = 0; k < kc; k++)
                            sum += pA[k * MC_MAX + ii] * pB[k * NC_MAX + jj];
                        float *c_ptr = C + (i0 + ii) * N + (j0 + jj);
                        if (k0 == 0) *c_ptr = alpha * sum + (beta_zero ? 0.0f : beta * *c_ptr);
                        else *c_ptr += alpha * sum;
                    }
                }
            }
        }
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        _mm_free(packA[tid]);
        _mm_free(packB[tid]);
    }
    free(packA);
    free(packB);
}
