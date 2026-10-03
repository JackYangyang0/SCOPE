#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MC = 64;
    const int NC = 256;
    const int KC = 256;
    const int MR = 8;
    const int NR = 8;

    int i, j, k, ii, jj, kk;
    int mc, nc, kc;
    int m_blocks = (M + MC - 1) / MC;
    int n_blocks = (N + NC - 1) / NC;
    int k_blocks = (K + KC - 1) / KC;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    for (kk = 0; kk < k_blocks; kk++) {
        kc = (kk == k_blocks - 1) ? (K - kk * KC) : KC;
        for (jj = 0; jj < n_blocks; jj++) {
            nc = (jj == n_blocks - 1) ? (N - jj * NC) : NC;
            int j_start = jj * NC;
            int j_end = j_start + nc;

            int num_threads = omp_get_max_threads();
            int chunk_size = (nc + num_threads - 1) / num_threads;
            if (chunk_size < NR) chunk_size = NR;
            int n_outer = (nc + chunk_size - 1) / chunk_size;

            int b;
            #pragma omp parallel for
            for (b = 0; b < n_outer; b++) {
                int j0 = j_start + b * chunk_size;
                int j1 = j0 + chunk_size;
                if (j1 > j_end) j1 = j_end;

                for (i = 0; i < M; i += MR) {
                    mc = (i + MR > M) ? (M - i) : MR;
                    for (j = j0; j < j1; j += NR) {
                        int nr = (j + NR > N) ? (N - j) : NR;

                        __m256 c0 = _mm256_setzero_ps();
                        __m256 c1 = _mm256_setzero_ps();
                        __m256 c2 = _mm256_setzero_ps();
                        __m256 c3 = _mm256_setzero_ps();
                        __m256 c4 = _mm256_setzero_ps();
                        __m256 c5 = _mm256_setzero_ps();
                        __m256 c6 = _mm256_setzero_ps();
                        __m256 c7 = _mm256_setzero_ps();

                        const float* a_ptr = &A[i * K + kk * KC];
                        const float* b_ptr = &B[(kk * KC) * N + j];
                        for (k = 0; k < kc; k++) {
                            __m256 a_vec = _mm256_loadu_ps(&a_ptr[k * K]);
                            if (nr >= 8) {
                                __m256 b_vec0 = _mm256_broadcast_ss(&b_ptr[k * N + 0]);
                                __m256 b_vec1 = _mm256_broadcast_ss(&b_ptr[k * N + 1]);
                                __m256 b_vec2 = _mm256_broadcast_ss(&b_ptr[k * N + 2]);
                                __m256 b_vec3 = _mm256_broadcast_ss(&b_ptr[k * N + 3]);
                                __m256 b_vec4 = _mm256_broadcast_ss(&b_ptr[k * N + 4]);
                                __m256 b_vec5 = _mm256_broadcast_ss(&b_ptr[k * N + 5]);
                                __m256 b_vec6 = _mm256_broadcast_ss(&b_ptr[k * N + 6]);
                                __m256 b_vec7 = _mm256_broadcast_ss(&b_ptr[k * N + 7]);
                                c0 = _mm256_fmadd_ps(a_vec, b_vec0, c0);
                                c1 = _mm256_fmadd_ps(a_vec, b_vec1, c1);
                                c2 = _mm256_fmadd_ps(a_vec, b_vec2, c2);
                                c3 = _mm256_fmadd_ps(a_vec, b_vec3, c3);
                                c4 = _mm256_fmadd_ps(a_vec, b_vec4, c4);
                                c5 = _mm256_fmadd_ps(a_vec, b_vec5, c5);
                                c6 = _mm256_fmadd_ps(a_vec, b_vec6, c6);
                                c7 = _mm256_fmadd_ps(a_vec, b_vec7, c7);
                            } else {
                                if (nr >= 1) {
                                    __m256 b_vec0 = _mm256_broadcast_ss(&b_ptr[k * N + 0]);
                                    c0 = _mm256_fmadd_ps(a_vec, b_vec0, c0);
                                }
                                if (nr >= 2) {
                                    __m256 b_vec1 = _mm256_broadcast_ss(&b_ptr[k * N + 1]);
                                    c1 = _mm256_fmadd_ps(a_vec, b_vec1, c1);
                                }
                                if (nr >= 3) {
                                    __m256 b_vec2 = _mm256_broadcast_ss(&b_ptr[k * N + 2]);
                                    c2 = _mm256_fmadd_ps(a_vec, b_vec2, c2);
                                }
                                if (nr >= 4) {
                                    __m256 b_vec3 = _mm256_broadcast_ss(&b_ptr[k * N + 3]);
                                    c3 = _mm256_fmadd_ps(a_vec, b_vec3, c3);
                                }
                                if (nr >= 5) {
                                    __m256 b_vec4 = _mm256_broadcast_ss(&b_ptr[k * N + 4]);
                                    c4 = _mm256_fmadd_ps(a_vec, b_vec4, c4);
                                }
                                if (nr >= 6) {
                                    __m256 b_vec5 = _mm256_broadcast_ss(&b_ptr[k * N + 5]);
                                    c5 = _mm256_fmadd_ps(a_vec, b_vec5, c5);
                                }
                                if (nr >= 7) {
                                    __m256 b_vec6 = _mm256_broadcast_ss(&b_ptr[k * N + 6]);
                                    c6 = _mm256_fmadd_ps(a_vec, b_vec6, c6);
                                }
                            }
                        }

                        float* c_store = &C[i * N + j];
                        if (nr >= 8) {
                            if (mc >= 8) {
                                _mm256_storeu_ps(&c_store[0 * N], c0);
                                _mm256_storeu_ps(&c_store[1 * N], c1);
                                _mm256_storeu_ps(&c_store[2 * N], c2);
                                _mm256_storeu_ps(&c_store[3 * N], c3);
                                _mm256_storeu_ps(&c_store[4 * N], c4);
                                _mm256_storeu_ps(&c_store[5 * N], c5);
                                _mm256_storeu_ps(&c_store[6 * N], c6);
                                _mm256_storeu_ps(&c_store[7 * N], c7);
                            } else {
                                float temp0[8], temp1[8], temp2[8], temp3[8];
                                float temp4[8], temp5[8], temp6[8], temp7[8];
                                _mm256_storeu_ps(temp0, c0);
                                _mm256_storeu_ps(temp1, c1);
                                _mm256_storeu_ps(temp2, c2);
                                _mm256_storeu_ps(temp3, c3);
                                _mm256_storeu_ps(temp4, c4);
                                _mm256_storeu_ps(temp5, c5);
                                _mm256_storeu_ps(temp6, c6);
                                _mm256_storeu_ps(temp7, c7);
                                for (ii = 0; ii < mc; ii++) {
                                    c_store[ii * N + 0] = temp0[ii];
                                    c_store[ii * N + 1] = temp1[ii];
                                    c_store[ii * N + 2] = temp2[ii];
                                    c_store[ii * N + 3] = temp3[ii];
                                    c_store[ii * N + 4] = temp4[ii];
                                    c_store[ii * N + 5] = temp5[ii];
                                    c_store[ii * N + 6] = temp6[ii];
                                    c_store[ii * N + 7] = temp7[ii];
                                }
                            }
                        } else {
                            float temp0[8], temp1[8], temp2[8], temp3[8];
                            float temp4[8], temp5[8], temp6[8], temp7[8];
                            _mm256_storeu_ps(temp0, c0);
                            _mm256_storeu_ps(temp1, c1);
                            _mm256_storeu_ps(temp2, c2);
                            _mm256_storeu_ps(temp3, c3);
                            _mm256_storeu_ps(temp4, c4);
                            _mm256_storeu_ps(temp5, c5);
                            _mm256_storeu_ps(temp6, c6);
                            _mm256_storeu_ps(temp7, c7);
                            for (ii = 0; ii < mc; ii++) {
                                if (nr >= 1) c_store[ii * N + 0] = temp0[ii];
                                if (nr >= 2) c_store[ii * N + 1] = temp1[ii];
                                if (nr >= 3) c_store[ii * N + 2] = temp2[ii];
                                if (nr >= 4) c_store[ii * N + 3] = temp3[ii];
                                if (nr >= 5) c_store[ii * N + 4] = temp4[ii];
                                if (nr >= 6) c_store[ii * N + 5] = temp5[ii];
                                if (nr >= 7) c_store[ii * N + 6] = temp6[ii];
                            }
                        }
                    }
                }
            }
        }
    }
}