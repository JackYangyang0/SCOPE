#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    // Optimized blocking parameters for multi-thread L2 cache residency
    // Target working set per thread: (MC + NC) * KC * 4 <= 192KB (conservative for 8+ threads)
    const int MC = 144;   // Multiple of MR=6
    const int NC = 192;   // Multiple of NR=16
    const int KC = 128;   // (144+192)*128*4 = 172KB < typical per-core L2 (2MB)
    const int MR = 6;
    const int NR = 16;

    int num_blocks_i = (M + MC - 1) / MC;
    int num_blocks_j = (N + NC - 1) / NC;

#pragma omp parallel
    {
        float* packedA = (float*)_mm_malloc(MC * KC * sizeof(float), 32);
        float* packedB = (float*)_mm_malloc(NC * KC * sizeof(float), 32);

        int b_i, b_j;
#pragma omp for collapse(2) schedule(static)
        for (b_i = 0; b_i < num_blocks_i; b_i++) {
            for (b_j = 0; b_j < num_blocks_j; b_j++) {
                int i = b_i * MC;
                int j = b_j * NC;
                int mc = (M - i) < MC ? (M - i) : MC;
                int nc = (N - j) < NC ? (N - j) : NC;

                for (int k = 0; k < K; k += KC) {
                    int kc = (K - k) < KC ? (K - k) : KC;

                    // Pack A: [mc x kc] in row-major order
                    for (int ii = 0; ii < mc; ii++) {
                        int kk_vec = 0;
                        for (; kk_vec <= kc - 8; kk_vec += 8) {
                            __m256 a_row = _mm256_loadu_ps(&A[(i + ii) * K + (k + kk_vec)]);
                            _mm256_store_ps(&packedA[ii * KC + kk_vec], a_row);
                        }
                        for (; kk_vec < kc; kk_vec++) {
                            packedA[ii * KC + kk_vec] = A[(i + ii) * K + (k + kk_vec)];
                        }
                    }

                    // Pack B: [kc x nc] -> [nc x kc] (transposed)
                    for (int jj = 0; jj < nc; jj++) {
                        int kk_vec = 0;
                        for (; kk_vec <= kc - 8; kk_vec += 8) {
                            __m128 b_low = _mm_loadu_ps(&B[(k + kk_vec + 0) * N + (j + jj)]);
                            __m128 b_high = _mm_loadu_ps(&B[(k + kk_vec + 4) * N + (j + jj)]);
                            __m256 b_col = _mm256_insertf128_ps(_mm256_castps128_ps256(b_low), b_high, 1);
                            _mm256_store_ps(&packedB[jj * KC + kk_vec], b_col);
                        }
                        for (; kk_vec < kc; kk_vec++) {
                            packedB[jj * KC + kk_vec] = B[(k + kk_vec) * N + (j + jj)];
                        }
                    }

                    // Micro-kernel: compute mc x nc block of C
                    for (int ii = 0; ii < mc; ii += MR) {
                        int mr = (mc - ii) < MR ? (mc - ii) : MR;
                        for (int jj = 0; jj < nc; jj += NR) {
                            int nr = (nc - jj) < NR ? (nc - jj) : NR;

                            // Initialize accumulators
                            __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
                            __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
                            __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
                            __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
                            __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
                            __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

                            for (int kk = 0; kk < kc; kk += 8) {
                                // Load A vectors (6 rows)
                                __m256 a0 = _mm256_load_ps(&packedA[(ii + 0) * KC + kk]);
                                __m256 a1 = _mm256_load_ps(&packedA[(ii + 1) * KC + kk]);
                                __m256 a2 = _mm256_load_ps(&packedA[(ii + 2) * KC + kk]);
                                __m256 a3 = _mm256_load_ps(&packedA[(ii + 3) * KC + kk]);
                                __m256 a4 = _mm256_load_ps(&packedA[(ii + 4) * KC + kk]);
                                __m256 a5 = _mm256_load_ps(&packedA[(ii + 5) * KC + kk]);

                                // Load B vectors (16 columns)
                                __m256 b0 = _mm256_load_ps(&packedB[(jj + 0) * KC + kk]);
                                __m256 b1 = _mm256_load_ps(&packedB[(jj + 1) * KC + kk]);
                                __m256 b2 = _mm256_load_ps(&packedB[(jj + 2) * KC + kk]);
                                __m256 b3 = _mm256_load_ps(&packedB[(jj + 3) * KC + kk]);
                                __m256 b4 = _mm256_load_ps(&packedB[(jj + 4) * KC + kk]);
                                __m256 b5 = _mm256_load_ps(&packedB[(jj + 5) * KC + kk]);
                                __m256 b6 = _mm256_load_ps(&packedB[(jj + 6) * KC + kk]);
                                __m256 b7 = _mm256_load_ps(&packedB[(jj + 7) * KC + kk]);
                                __m256 b8 = _mm256_load_ps(&packedB[(jj + 8) * KC + kk]);
                                __m256 b9 = _mm256_load_ps(&packedB[(jj + 9) * KC + kk]);
                                __m256 b10 = _mm256_load_ps(&packedB[(jj + 10) * KC + kk]);
                                __m256 b11 = _mm256_load_ps(&packedB[(jj + 11) * KC + kk]);
                                __m256 b12 = _mm256_load_ps(&packedB[(jj + 12) * KC + kk]);
                                __m256 b13 = _mm256_load_ps(&packedB[(jj + 13) * KC + kk]);
                                __m256 b14 = _mm256_load_ps(&packedB[(jj + 14) * KC + kk]);
                                __m256 b15 = _mm256_load_ps(&packedB[(jj + 15) * KC + kk]);

                                // Interleaved FMA operations for better ILP
                                c00 = _mm256_fmadd_ps(a0, b0, c00);
                                c01 = _mm256_fmadd_ps(a0, b8, c01);
                                c10 = _mm256_fmadd_ps(a1, b0, c10);
                                c11 = _mm256_fmadd_ps(a1, b8, c11);
                                c20 = _mm256_fmadd_ps(a2, b0, c20);
                                c21 = _mm256_fmadd_ps(a2, b8, c21);
                                c30 = _mm256_fmadd_ps(a3, b0, c30);
                                c31 = _mm256_fmadd_ps(a3, b8, c31);
                                c40 = _mm256_fmadd_ps(a4, b0, c40);
                                c41 = _mm256_fmadd_ps(a4, b8, c41);
                                c50 = _mm256_fmadd_ps(a5, b0, c50);
                                c51 = _mm256_fmadd_ps(a5, b8, c51);

                                c00 = _mm256_fmadd_ps(a0, b1, c00);
                                c01 = _mm256_fmadd_ps(a0, b9, c01);
                                c10 = _mm256_fmadd_ps(a1, b1, c10);
                                c11 = _mm256_fmadd_ps(a1, b9, c11);
                                c20 = _mm256_fmadd_ps(a2, b1, c20);
                                c21 = _mm256_fmadd_ps(a2, b9, c21);
                                c30 = _mm256_fmadd_ps(a3, b1, c30);
                                c31 = _mm256_fmadd_ps(a3, b9, c31);
                                c40 = _mm256_fmadd_ps(a4, b1, c40);
                                c41 = _mm256_fmadd_ps(a4, b9, c41);
                                c50 = _mm256_fmadd_ps(a5, b1, c50);
                                c51 = _mm256_fmadd_ps(a5, b9, c51);

                                c00 = _mm256_fmadd_ps(a0, b2, c00);
                                c01 = _mm256_fmadd_ps(a0, b10, c01);
                                c10 = _mm256_fmadd_ps(a1, b2, c10);
                                c11 = _mm256_fmadd_ps(a1, b10, c11);
                                c20 = _mm256_fmadd_ps(a2, b2, c20);
                                c21 = _mm256_fmadd_ps(a2, b10, c21);
                                c30 = _mm256_fmadd_ps(a3, b2, c30);
                                c31 = _mm256_fmadd_ps(a3, b10, c31);
                                c40 = _mm256_fmadd_ps(a4, b2, c40);
                                c41 = _mm256_fmadd_ps(a4, b10, c41);
                                c50 = _mm256_fmadd_ps(a5, b2, c50);
                                c51 = _mm256_fmadd_ps(a5, b10, c51);

                                c00 = _mm256_fmadd_ps(a0, b3, c00);
                                c01 = _mm256_fmadd_ps(a0, b11, c01);
                                c10 = _mm256_fmadd_ps(a1, b3, c10);
                                c11 = _mm256_fmadd_ps(a1, b11, c11);
                                c20 = _mm256_fmadd_ps(a2, b3, c20);
                                c21 = _mm256_fmadd_ps(a2, b11, c21);
                                c30 = _mm256_fmadd_ps(a3, b3, c30);
                                c31 = _mm256_fmadd_ps(a3, b11, c31);
                                c40 = _mm256_fmadd_ps(a4, b3, c40);
                                c41 = _mm256_fmadd_ps(a4, b11, c41);
                                c50 = _mm256_fmadd_ps(a5, b3, c50);
                                c51 = _mm256_fmadd_ps(a5, b11, c51);

                                c00 = _mm256_fmadd_ps(a0, b4, c00);
                                c01 = _mm256_fmadd_ps(a0, b12, c01);
                                c10 = _mm256_fmadd_ps(a1, b4, c10);
                                c11 = _mm256_fmadd_ps(a1, b12, c11);
                                c20 = _mm256_fmadd_ps(a2, b4, c20);
                                c21 = _mm256_fmadd_ps(a2, b12, c21);
                                c30 = _mm256_fmadd_ps(a3, b4, c30);
                                c31 = _mm256_fmadd_ps(a3, b12, c31);
                                c40 = _mm256_fmadd_ps(a4, b4, c40);
                                c41 = _mm256_fmadd_ps(a4, b12, c41);
                                c50 = _mm256_fmadd_ps(a5, b4, c50);
                                c51 = _mm256_fmadd_ps(a5, b12, c51);

                                c00 = _mm256_fmadd_ps(a0, b5, c00);
                                c01 = _mm256_fmadd_ps(a0, b13, c01);
                                c10 = _mm256_fmadd_ps(a1, b5, c10);
                                c11 = _mm256_fmadd_ps(a1, b13, c11);
                                c20 = _mm256_fmadd_ps(a2, b5, c20);
                                c21 = _mm256_fmadd_ps(a2, b13, c21);
                                c30 = _mm256_fmadd_ps(a3, b5, c30);
                                c31 = _mm256_fmadd_ps(a3, b13, c31);
                                c40 = _mm256_fmadd_ps(a4, b5, c40);
                                c41 = _mm256_fmadd_ps(a4, b13, c41);
                                c50 = _mm256_fmadd_ps(a5, b5, c50);
                                c51 = _mm256_fmadd_ps(a5, b13, c51);

                                c00 = _mm256_fmadd_ps(a0, b6, c00);
                                c01 = _mm256_fmadd_ps(a0, b14, c01);
                                c10 = _mm256_fmadd_ps(a1, b6, c10);
                                c11 = _mm256_fmadd_ps(a1, b14, c11);
                                c20 = _mm256_fmadd_ps(a2, b6, c20);
                                c21 = _mm256_fmadd_ps(a2, b14, c21);
                                c30 = _mm256_fmadd_ps(a3, b6, c30);
                                c31 = _mm256_fmadd_ps(a3, b14, c31);
                                c40 = _mm256_fmadd_ps(a4, b6, c40);
                                c41 = _mm256_fmadd_ps(a4, b14, c41);
                                c50 = _mm256_fmadd_ps(a5, b6, c50);
                                c51 = _mm256_fmadd_ps(a5, b14, c51);

                                c00 = _mm256_fmadd_ps(a0, b7, c00);
                                c01 = _mm256_fmadd_ps(a0, b15, c01);
                                c10 = _mm256_fmadd_ps(a1, b7, c10);
                                c11 = _mm256_fmadd_ps(a1, b15, c11);
                                c20 = _mm256_fmadd_ps(a2, b7, c20);
                                c21 = _mm256_fmadd_ps(a2, b15, c21);
                                c30 = _mm256_fmadd_ps(a3, b7, c30);
                                c31 = _mm256_fmadd_ps(a3, b15, c31);
                                c40 = _mm256_fmadd_ps(a4, b7, c40);
                                c41 = _mm256_fmadd_ps(a4, b15, c41);
                                c50 = _mm256_fmadd_ps(a5, b7, c50);
                                c51 = _mm256_fmadd_ps(a5, b15, c51);
                            }

                            // Store results back to C
                            if (nr >= 8) {
                                if (mr >= 1) _mm256_storeu_ps(&C[(i + ii + 0) * N + (j + jj + 0)], c00);
                                if (mr >= 2) _mm256_storeu_ps(&C[(i + ii + 1) * N + (j + jj + 0)], c10);
                                if (mr >= 3) _mm256_storeu_ps(&C[(i + ii + 2) * N + (j + jj + 0)], c20);
                                if (mr >= 4) _mm256_storeu_ps(&C[(i + ii + 3) * N + (j + jj + 0)], c30);
                                if (mr >= 5) _mm256_storeu_ps(&C[(i + ii + 4) * N + (j + jj + 0)], c40);
                                if (mr >= 6) _mm256_storeu_ps(&C[(i + ii + 5) * N + (j + jj + 0)], c50);
                                
                                if (nr >= 16) {
                                    if (mr >= 1) _mm256_storeu_ps(&C[(i + ii + 0) * N + (j + jj + 8)], c01);
                                    if (mr >= 2) _mm256_storeu_ps(&C[(i + ii + 1) * N + (j + jj + 8)], c11);
                                    if (mr >= 3) _mm256_storeu_ps(&C[(i + ii + 2) * N + (j + jj + 8)], c21);
                                    if (mr >= 4) _mm256_storeu_ps(&C[(i + ii + 3) * N + (j + jj + 8)], c31);
                                    if (mr >= 5) _mm256_storeu_ps(&C[(i + ii + 4) * N + (j + jj + 8)], c41);
                                    if (mr >= 6) _mm256_storeu_ps(&C[(i + ii + 5) * N + (j + jj + 8)], c51);
                                } else {
                                    int rem = nr - 8;
                                    if (mr >= 1) {
                                        float* cptr = &C[(i + ii + 0) * N + (j + jj + 8)];
                                        for (int r = 0; r < rem; r++) cptr[r] = ((float*)&c01)[r];
                                    }
                                    if (mr >= 2) {
                                        float* cptr = &C[(i + ii + 1) * N + (j + jj + 8)];
                                        for (int r = 0; r < rem; r++) cptr[r] = ((float*)&c11)[r];
                                    }
                                    if (mr >= 3) {
                                        float* cptr = &C[(i + ii + 2) * N + (j + jj + 8)];
                                        for (int r = 0; r < rem; r++) cptr[r] = ((float*)&c21)[r];
                                    }
                                    if (mr >= 4) {
                                        float* cptr = &C[(i + ii + 3) * N + (j + jj + 8)];
                                        for (int r = 0; r < rem; r++) cptr[r] = ((float*)&c31)[r];
                                    }
                                    if (mr >= 5) {
                                        float* cptr = &C[(i + ii + 4) * N + (j + jj + 8)];
                                        for (int r = 0; r < rem; r++) cptr[r] = ((float*)&c41)[r];
                                    }
                                    if (mr >= 6) {
                                        float* cptr = &C[(i + ii + 5) * N + (j + jj + 8)];
                                        for (int r = 0; r < rem; r++) cptr[r] = ((float*)&c51)[r];
                                    }
                                }
                            } else {
                                for (int rj = 0; rj < nr; rj++) {
                                    if (mr >= 1) C[(i + ii + 0) * N + (j + jj + rj)] = ((float*)&c00)[rj];
                                    if (mr >= 2) C[(i + ii + 1) * N + (j + jj + rj)] = ((float*)&c10)[rj];
                                    if (mr >= 3) C[(i + ii + 2) * N + (j + jj + rj)] = ((float*)&c20)[rj];
                                    if (mr >= 4) C[(i + ii + 3) * N + (j + jj + rj)] = ((float*)&c30)[rj];
                                    if (mr >= 5) C[(i + ii + 4) * N + (j + jj + rj)] = ((float*)&c40)[rj];
                                    if (mr >= 6) C[(i + ii + 5) * N + (j + jj + rj)] = ((float*)&c50)[rj];
                                }
                            }
                        }
                    }
                }
            }
        }
        _mm_free(packedA);
        _mm_free(packedB);
    }
}