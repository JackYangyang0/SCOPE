#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MC = 128;
    const int NC = 4096;
    const int KC = 256;
    const int MR = 8;
    const int NR = 16;

    int nthreads = omp_get_max_threads();
    size_t packedA_stride = (size_t)MC * KC;
    size_t packedB_stride = (size_t)KC * NC;
    size_t packedA_size = (packedA_stride + 7) & ~7;
    size_t packedB_size = (packedB_stride + 7) & ~7;
    float* all_packedA = (float*)_mm_malloc(nthreads * packedA_size * sizeof(float), 64);
    float* all_packedB = (float*)_mm_malloc(nthreads * packedB_size * sizeof(float), 64);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* packedA = &all_packedA[tid * packedA_size];
        float* packedB = &all_packedB[tid * packedB_size];

        int ib;
        #pragma omp for schedule(static)
        for (ib = 0; ib < M; ib += MC) {
            int mc = (M - ib < MC) ? M - ib : MC;

            for (int kb = 0; kb < K; kb += KC) {
                int kc = (K - kb < KC) ? K - kb : KC;

                // Pack A: mc x kc -> mc x KC (row-major)
                for (int i = 0; i < mc; i++) {
                    const float* a_row = &A[(ib + i) * K + kb];
                    float* pa_base = &packedA[i * KC];
                    int k = 0;
                    for (; k < kc; k++) {
                        pa_base[k] = a_row[k];
                    }
                    for (; k < KC; k++) {
                        pa_base[k] = 0.0f;
                    }
                }

                // Pack B: kc x N -> KC x min(NC, remaining N)
                int nc_total = N;
                for (int jb = 0; jb < nc_total; jb += NC) {
                    int nc = (nc_total - jb < NC) ? nc_total - jb : NC;

                    for (int k = 0; k < kc; k++) {
                        const float* b_row = &B[(kb + k) * N + jb];
                        float* pb_base = &packedB[k * NC];
                        int j = 0;
                        for (; j < nc; j++) {
                            pb_base[j] = b_row[j];
                        }
                        for (; j < NC; j++) {
                            pb_base[j] = 0.0f;
                        }
                    }
                    for (int k = kc; k < KC; k++) {
                        float* pb_base = &packedB[k * NC];
                        for (int j = 0; j < NC; j++) {
                            pb_base[j] = 0.0f;
                        }
                    }

                    // Microkernel: mc x nc
                    for (int i = 0; i < mc; i += MR) {
                        int mr = (mc - i < MR) ? mc - i : MR;
                        for (int j = 0; j < nc; j += NR) {
                            int nr = (nc - j < NR) ? nc - j : NR;

                            __m256 c_vec[MR][2];
                            for (int ii = 0; ii < mr; ii++) {
                                c_vec[ii][0] = _mm256_setzero_ps();
                                c_vec[ii][1] = _mm256_setzero_ps();
                            }

                            for (int kk = 0; kk < KC; kk++) {
                                __m256 b0 = _mm256_loadu_ps(&packedB[kk * NC + j]);
                                __m256 b1 = _mm256_loadu_ps(&packedB[kk * NC + j + 8]);
                                for (int ii = 0; ii < mr; ii++) {
                                    float a_val = packedA[(i + ii) * KC + kk];
                                    __m256 a_vec = _mm256_set1_ps(a_val);
                                    c_vec[ii][0] = _mm256_fmadd_ps(a_vec, b0, c_vec[ii][0]);
                                    c_vec[ii][1] = _mm256_fmadd_ps(a_vec, b1, c_vec[ii][1]);
                                }
                            }

                            for (int ii = 0; ii < mr; ii++) {
                                if (nr >= 16) {
                                    _mm256_storeu_ps(&C[(ib + i + ii) * N + jb + j], c_vec[ii][0]);
                                    _mm256_storeu_ps(&C[(ib + i + ii) * N + jb + j + 8], c_vec[ii][1]);
                                } else if (nr > 8) {
                                    _mm256_storeu_ps(&C[(ib + i + ii) * N + jb + j], c_vec[ii][0]);
                                    __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr - 8), _mm256_set_epi32(7,6,5,4,3,2,1,0));
                                    _mm256_maskstore_ps(&C[(ib + i + ii) * N + jb + j + 8], mask, c_vec[ii][1]);
                                } else {
                                    __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr), _mm256_set_epi32(7,6,5,4,3,2,1,0));
                                    _mm256_maskstore_ps(&C[(ib + i + ii) * N + jb + j], mask, c_vec[ii][0]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    _mm_free(all_packedA);
    _mm_free(all_packedB);
}