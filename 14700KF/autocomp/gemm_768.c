#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MC = 256;
    const int NC = 256;
    const int KC = 256;
    const int MR = 8;
    const int NR = 8;

    int i, j, k, ii, jj, kk;
    int mc, nc, kc;
    int mr, nr;

    // Zero out C matrix
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    for (j = 0; j < N; j += NC) {
        nc = N - j < NC ? N - j : NC;

        #pragma omp parallel for schedule(static)
        for (i = 0; i < M; i += MC) {
            mc = M - i < MC ? M - i : MC;

            for (k = 0; k < K; k += KC) {
                kc = K - k < KC ? K - k : KC;

                // Pack B panel: [kc x nc] -> [nc x kc] with contiguous rows
                float* B_panel = (float*)(((uintptr_t)B + 63) & ~63); // Not actually used - we'll allocate properly
                // Instead, we'll use a local packed buffer on stack (aligned)
                float packed_B[NC * KC + 8]; // +8 for alignment padding
                float* B_packed = (float*)(((uintptr_t)packed_B + 31) & ~31);

                // Pack B[k:k+kc, j:j+nc] into B_packed in row-major order (so each row of B becomes contiguous)
                for (jj = 0; jj < nc; jj++) {
                    for (kk = 0; kk < kc; kk++) {
                        B_packed[jj * kc + kk] = B[(k + kk) * N + j + jj];
                    }
                }

                // Micro-kernel: compute C[i:i+mc, j:j+nc] += A[i:i+mc, k:k+kc] * B_packed[0:nc, 0:kc]
                for (ii = 0; ii < mc; ii += MR) {
                    mr = mc - ii < MR ? mc - ii : MR;
                    for (jj = 0; jj < nc; jj += NR) {
                        nr = nc - jj < NR ? nc - jj : NR;

                        // Initialize accumulator registers
                        __m256 c0 = _mm256_setzero_ps();
                        __m256 c1 = _mm256_setzero_ps();
                        __m256 c2 = _mm256_setzero_ps();
                        __m256 c3 = _mm256_setzero_ps();
                        __m256 c4 = _mm256_setzero_ps();
                        __m256 c5 = _mm256_setzero_ps();
                        __m256 c6 = _mm256_setzero_ps();
                        __m256 c7 = _mm256_setzero_ps();

                        // Main loop over k dimension
                        for (kk = 0; kk < kc; kk++) {
                            const float* a_ptr = &A[(i + ii) * K + k + kk];
                            
                            __m256 b_vec;
                            if (nr >= 8) {
                                b_vec = _mm256_loadu_ps(&B_packed[jj * kc + kk]);
                                
                                if (mr > 0) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[0]);
                                    c0 = _mm256_fmadd_ps(a_val, b_vec, c0);
                                }
                                if (mr > 1) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[K]);
                                    c1 = _mm256_fmadd_ps(a_val, b_vec, c1);
                                }
                                if (mr > 2) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[2*K]);
                                    c2 = _mm256_fmadd_ps(a_val, b_vec, c2);
                                }
                                if (mr > 3) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[3*K]);
                                    c3 = _mm256_fmadd_ps(a_val, b_vec, c3);
                                }
                                if (mr > 4) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[4*K]);
                                    c4 = _mm256_fmadd_ps(a_val, b_vec, c4);
                                }
                                if (mr > 5) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[5*K]);
                                    c5 = _mm256_fmadd_ps(a_val, b_vec, c5);
                                }
                                if (mr > 6) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[6*K]);
                                    c6 = _mm256_fmadd_ps(a_val, b_vec, c6);
                                }
                                if (mr > 7) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[7*K]);
                                    c7 = _mm256_fmadd_ps(a_val, b_vec, c7);
                                }
                            } else {
                                // Handle smaller nr cases
                                float b_temp[8] = {0};
                                for (int idx = 0; idx < nr; idx++) {
                                    b_temp[idx] = B_packed[(jj + idx) * kc + kk];
                                }
                                b_vec = _mm256_loadu_ps(b_temp);
                                
                                if (mr > 0) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[0]);
                                    c0 = _mm256_fmadd_ps(a_val, b_vec, c0);
                                }
                                if (mr > 1) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[K]);
                                    c1 = _mm256_fmadd_ps(a_val, b_vec, c1);
                                }
                                if (mr > 2) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[2*K]);
                                    c2 = _mm256_fmadd_ps(a_val, b_vec, c2);
                                }
                                if (mr > 3) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[3*K]);
                                    c3 = _mm256_fmadd_ps(a_val, b_vec, c3);
                                }
                                if (mr > 4) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[4*K]);
                                    c4 = _mm256_fmadd_ps(a_val, b_vec, c4);
                                }
                                if (mr > 5) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[5*K]);
                                    c5 = _mm256_fmadd_ps(a_val, b_vec, c5);
                                }
                                if (mr > 6) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[6*K]);
                                    c6 = _mm256_fmadd_ps(a_val, b_vec, c6);
                                }
                                if (mr > 7) {
                                    __m256 a_val = _mm256_broadcast_ss(&a_ptr[7*K]);
                                    c7 = _mm256_fmadd_ps(a_val, b_vec, c7);
                                }
                            }
                        }

                        // Store results back to C
                        float* c_ptr = &C[(i + ii) * N + j + jj];
                        if (nr >= 8) {
                            if (mr > 0) _mm256_storeu_ps(&c_ptr[0], c0);
                            if (mr > 1) _mm256_storeu_ps(&c_ptr[N], c1);
                            if (mr > 2) _mm256_storeu_ps(&c_ptr[2*N], c2);
                            if (mr > 3) _mm256_storeu_ps(&c_ptr[3*N], c3);
                            if (mr > 4) _mm256_storeu_ps(&c_ptr[4*N], c4);
                            if (mr > 5) _mm256_storeu_ps(&c_ptr[5*N], c5);
                            if (mr > 6) _mm256_storeu_ps(&c_ptr[6*N], c6);
                            if (mr > 7) _mm256_storeu_ps(&c_ptr[7*N], c7);
                        } else {
                            float temp[8];
                            if (mr > 0) {
                                _mm256_storeu_ps(temp, c0);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[idx] = temp[idx];
                                }
                            }
                            if (mr > 1) {
                                _mm256_storeu_ps(temp, c1);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[N + idx] = temp[idx];
                                }
                            }
                            if (mr > 2) {
                                _mm256_storeu_ps(temp, c2);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[2*N + idx] = temp[idx];
                                }
                            }
                            if (mr > 3) {
                                _mm256_storeu_ps(temp, c3);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[3*N + idx] = temp[idx];
                                }
                            }
                            if (mr > 4) {
                                _mm256_storeu_ps(temp, c4);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[4*N + idx] = temp[idx];
                                }
                            }
                            if (mr > 5) {
                                _mm256_storeu_ps(temp, c5);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[5*N + idx] = temp[idx];
                                }
                            }
                            if (mr > 6) {
                                _mm256_storeu_ps(temp, c6);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[6*N + idx] = temp[idx];
                                }
                            }
                            if (mr > 7) {
                                _mm256_storeu_ps(temp, c7);
                                for (int idx = 0; idx < nr; idx++) {
                                    c_ptr[7*N + idx] = temp[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}