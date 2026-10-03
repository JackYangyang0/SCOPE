#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MR = 16;
    const int NR = 32;

    int i, j;
    #pragma omp parallel for collapse(2) schedule(static)
    for (i = 0; i < M; i += MR) {
        for (j = 0; j < N; j += NR) {
            int i_end = (i + MR > M) ? M : i + MR;
            int j_end = (j + NR > N) ? N : j + NR;
            int mr = i_end - i;
            int nr = j_end - j;

            __m256 c_reg[2][4];
            int ii, jj;
            for (ii = 0; ii < 2; ii++) {
                for (jj = 0; jj < 4; jj++) {
                    c_reg[ii][jj] = _mm256_setzero_ps();
                }
            }

            int k;
            for (k = 0; k < K; k++) {
                __m256 a_vals[2];
                if (mr > 8) {
                    a_vals[0] = _mm256_set1_ps(A[(i + 0) * K + k]);
                    a_vals[1] = _mm256_set1_ps(A[(i + 8) * K + k]);
                } else {
                    a_vals[0] = _mm256_set1_ps(A[(i + 0) * K + k]);
                    a_vals[1] = _mm256_setzero_ps();
                }

                __m256 b_vals[4];
                for (jj = 0; jj < 4; jj++) {
                    if (j + jj * 8 < j_end) {
                        float b_scalar = B[k * N + j + jj * 8];
                        b_vals[jj] = _mm256_set1_ps(b_scalar);
                    } else {
                        b_vals[jj] = _mm256_setzero_ps();
                    }
                }

                for (ii = 0; ii < 2; ii++) {
                    for (jj = 0; jj < 4; jj++) {
                        c_reg[ii][jj] = _mm256_fmadd_ps(a_vals[ii], b_vals[jj], c_reg[ii][jj]);
                    }
                }
            }

            for (ii = 0; ii < mr; ii++) {
                for (jj = 0; jj < nr; jj++) {
                    int vec_i = ii / 8;
                    int lane_i = ii % 8;
                    int vec_j = jj / 8;
                    int lane_j = jj % 8;
                    float* c_ptr = (float*)&c_reg[vec_i][vec_j];
                    C[(i + ii) * N + j + jj] = c_ptr[lane_i * 8 + lane_j];
                }
            }
        }
    }
}