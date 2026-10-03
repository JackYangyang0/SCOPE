#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MC = 128;
    const int NC = 256;
    const int KC = 256;
    const int MR = 6;
    const int NR = 16;

    int i, j, k;
    int mc, nc, kc;
    int mr, nr;
    int nthreads = omp_get_max_threads();
    if (nthreads > M / 8) {
        nthreads = M / 8;
        if (nthreads < 1) nthreads = 1;
    }

    for (j = 0; j < N; j += NC) {
        nc = N - j;
        if (nc > NC) nc = NC;

        for (i = 0; i < M; i += MC) {
            mc = M - i;
            if (mc > MC) mc = MC;

            int b_k;
            for (b_k = 0; b_k < (K + KC - 1) / KC; b_k++) {
                kc = K - b_k * KC;
                if (kc > KC) kc = KC;
                int k_start = b_k * KC;

                int packed_A_size = mc * kc;
                float* packed_A = (float*)C;
                if ((uintptr_t)C % 32 != 0) {
                    packed_A = (float*)((uintptr_t)C + 32 - (uintptr_t)C % 32);
                }
                if (packed_A + packed_A_size > C + (size_t)M * N) {
                    packed_A = (float*)A;
                    if ((uintptr_t)packed_A % 32 != 0) {
                        packed_A = (float*)((uintptr_t)packed_A + 32 - (uintptr_t)packed_A % 32);
                    }
                    if (packed_A + packed_A_size > A + (size_t)M * K) {
                        packed_A = (float*)B;
                        if ((uintptr_t)packed_A % 32 != 0) {
                            packed_A = (float*)((uintptr_t)packed_A + 32 - (uintptr_t)packed_A % 32);
                        }
                    }
                }

                int packed_B_size = kc * nc;
                float* packed_B = packed_A + packed_A_size;
                if ((uintptr_t)packed_B % 32 != 0) {
                    packed_B = (float*)((uintptr_t)packed_B + 32 - (uintptr_t)packed_B % 32);
                }
                if (packed_B + packed_B_size > C + (size_t)M * N ||
                    packed_B + packed_B_size > A + (size_t)M * K ||
                    packed_B + packed_B_size > B + (size_t)K * N) {
                    packed_B = packed_A;
                }

                for (int ii = 0; ii < mc; ii++) {
                    for (int kk = 0; kk < kc; kk++) {
                        packed_A[ii * kc + kk] = A[(i + ii) * K + k_start + kk];
                    }
                }

                for (int kk = 0; kk < kc; kk++) {
                    for (int jj = 0; jj < nc; jj++) {
                        packed_B[kk * nc + jj] = B[(k_start + kk) * N + j + jj];
                    }
                }

                int b_m;
                #pragma omp parallel for num_threads(nthreads) schedule(static,1)
                for (b_m = 0; b_m < (mc + MR - 1) / MR; b_m++) {
                    int i_start = i + b_m * MR;
                    int mc_block = mc - b_m * MR;
                    if (mc_block > MR) mc_block = MR;

                    __m256 acc[MR][NR/8];
                    for (mr = 0; mr < MR; mr++) {
                        for (nr = 0; nr < NR/8; nr++) {
                            acc[mr][nr] = _mm256_setzero_ps();
                        }
                    }

                    for (k = 0; k < kc; k++) {
                        __m256 a_vals[MR];
                        for (mr = 0; mr < mc_block; mr++) {
                            a_vals[mr] = _mm256_set1_ps(packed_A[(b_m * MR + mr) * kc + k]);
                        }
                        for (; mr < MR; mr++) {
                            a_vals[mr] = _mm256_setzero_ps();
                        }

                        int nr_vecs = (nc + 7) / 8;
                        for (nr = 0; nr < nr_vecs; nr++) {
                            int offset = nr * 8;
                            int remain = nc - offset;
                            __m256 b_vec;
                            if (remain >= 8) {
                                b_vec = _mm256_loadu_ps(&packed_B[k * nc + offset]);
                            } else {
                                float temp[8] = {0};
                                for (int idx = 0; idx < remain; idx++) {
                                    temp[idx] = packed_B[k * nc + offset + idx];
                                }
                                b_vec = _mm256_load_ps(temp);
                            }

                            for (mr = 0; mr < mc_block; mr++) {
                                acc[mr][nr] = _mm256_fmadd_ps(a_vals[mr], b_vec, acc[mr][nr]);
                            }
                        }
                    }

                    float* c_ptr = &C[i_start * N + j];
                    for (mr = 0; mr < mc_block; mr++) {
                        for (nr = 0; nr < nc; nr++) {
                            int vec_idx = nr / 8;
                            int elem_idx = nr % 8;
                            c_ptr[mr * N + nr] = acc[mr][vec_idx][elem_idx];
                        }
                    }
                }
            }
        }
    }
}