#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MC = 256;
    const int NC = 512;
    const int KC = 256;
    const int MR = 8;
    const int NR = 8;

    int m_blocks = (M + MC - 1) / MC;
    int n_blocks = (N + NC - 1) / NC;
    int k_blocks = (K + KC - 1) / KC;

    int i, j, k, ii, jj, kk, l;
    int thread_id, num_threads;
    float* packed_B;
    float* packed_A;
    int mc, nc, kc;
    int mr_rem, nr_rem;
    int a_index, b_index, c_index;
    __m256 a_vec, b_vec, c0, c1, c2, c3, c4, c5, c6, c7;
    int jjj;

    num_threads = omp_get_max_threads();
    packed_B = (float*)(((uintptr_t)C + M * N * sizeof(float) + 63) & (~63));
    packed_A = packed_B + NC * KC;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    for (k = 0; k < k_blocks; k++) {
        kc = (k == k_blocks - 1) ? (K - k * KC) : KC;
        if (kc <= 0) continue;

        for (j = 0; j < n_blocks; j++) {
            nc = (j == n_blocks - 1) ? (N - j * NC) : NC;
            if (nc <= 0) continue;

            float* pb = packed_B;
            const float* b_panel = B + k * KC * N + j * NC;
            for (l = 0; l < kc; l++) {
                for (jj = 0; jj < nc; jj++) {
                    pb[l * nc + jj] = b_panel[l * N + jj];
                }
            }

            int i_block;
            #pragma omp parallel for private(i, ii, jj, jjj, kk, a_index, b_index, c_index, \
                                            mr_rem, nr_rem, a_vec, b_vec, c0, c1, c2, c3, c4, c5, c6, c7)
            for (i_block = 0; i_block < m_blocks; i_block++) {
                mc = (i_block == m_blocks - 1) ? (M - i_block * MC) : MC;
                if (mc <= 0) continue;

                const float* a_block = A + i_block * MC * K + k * KC;
                float* c_block = C + i_block * MC * N + j * NC;

                for (i = 0; i < mc; i += MR) {
                    mr_rem = (mc - i < MR) ? (mc - i) : MR;
                    for (jj = 0; jj < nc; jj += NR) {
                        nr_rem = (nc - jj < NR) ? (nc - jj) : NR;

                        __m256 accumulators[MR][NR];
                        for (ii = 0; ii < MR; ii++) {
                            for (jjj = 0; jjj < NR; jjj++) {
                                accumulators[ii][jjj] = _mm256_setzero_ps();
                            }
                        }

                        for (kk = 0; kk < kc; kk++) {
                            if (mr_rem == MR) {
                                a_vec = _mm256_loadu_ps(&a_block[kk * K + i]);
                            } else {
                                float a_temp[MR];
                                for (ii = 0; ii < MR; ii++) {
                                    a_temp[ii] = (ii < mr_rem) ? a_block[kk * K + i + ii] : 0.0f;
                                }
                                a_vec = _mm256_loadu_ps(a_temp);
                            }

                            b_index = kk * nc + jj;
                            if (nr_rem == NR) {
                                b_vec = _mm256_loadu_ps(&packed_B[b_index]);
                            } else {
                                float b_temp[NR];
                                for (jjj = 0; jjj < NR; jjj++) {
                                    b_temp[jjj] = (jjj < nr_rem) ? packed_B[b_index + jjj] : 0.0f;
                                }
                                b_vec = _mm256_loadu_ps(b_temp);
                            }

                            c0 = accumulators[0][0]; c1 = accumulators[0][1]; c2 = accumulators[0][2]; c3 = accumulators[0][3];
                            c4 = accumulators[0][4]; c5 = accumulators[0][5]; c6 = accumulators[0][6]; c7 = accumulators[0][7];
                            c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(a_vec)), b_vec, c0);
                            c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0x55))), b_vec, c1);
                            c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0xAA))), b_vec, c2);
                            c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0xFF))), b_vec, c3);
                            
                            __m256 a_vec_high = _mm256_permute2f128_ps(a_vec, a_vec, 0x11);
                            c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(a_vec_high)), b_vec, c4);
                            c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0x55))), b_vec, c5);
                            c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xAA))), b_vec, c6);
                            c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c7);

                            accumulators[0][0] = c0; accumulators[0][1] = c1; accumulators[0][2] = c2; accumulators[0][3] = c3;
                            accumulators[0][4] = c4; accumulators[0][5] = c5; accumulators[0][6] = c6; accumulators[0][7] = c7;
                            
                            if (mr_rem > 1) {
                                c0 = accumulators[1][0]; c1 = accumulators[1][1]; c2 = accumulators[1][2]; c3 = accumulators[1][3];
                                c4 = accumulators[1][4]; c5 = accumulators[1][5]; c6 = accumulators[1][6]; c7 = accumulators[1][7];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0x55))), b_vec, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0xAA))), b_vec, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0xFF))), b_vec, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec, a_vec, 0x11))), b_vec, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0x55))), b_vec, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xAA))), b_vec, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11))), b_vec, c7);
                                
                                accumulators[1][0] = c0; accumulators[1][1] = c1; accumulators[1][2] = c2; accumulators[1][3] = c3;
                                accumulators[1][4] = c4; accumulators[1][5] = c5; accumulators[1][6] = c6; accumulators[1][7] = c7;
                            }
                            if (mr_rem > 2) {
                                c0 = accumulators[2][0]; c1 = accumulators[2][1]; c2 = accumulators[2][2]; c3 = accumulators[2][3];
                                c4 = accumulators[2][4]; c5 = accumulators[2][5]; c6 = accumulators[2][6]; c7 = accumulators[2][7];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0xAA))), b_vec, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0xFF))), b_vec, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec, a_vec, 0x11))), b_vec, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0x55))), b_vec, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xAA))), b_vec, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11))), b_vec, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec, a_vec, 0x11), 0x55))), b_vec, c7);
                                
                                accumulators[2][0] = c0; accumulators[2][1] = c1; accumulators[2][2] = c2; accumulators[2][3] = c3;
                                accumulators[2][4] = c4; accumulators[2][5] = c5; accumulators[2][6] = c6; accumulators[2][7] = c7;
                            }
                            if (mr_rem > 3) {
                                c0 = accumulators[3][0]; c1 = accumulators[3][1]; c2 = accumulators[3][2]; c3 = accumulators[3][3];
                                c4 = accumulators[3][4]; c5 = accumulators[3][5]; c6 = accumulators[3][6]; c7 = accumulators[3][7];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec, 0xFF))), b_vec, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec, a_vec, 0x11))), b_vec, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0x55))), b_vec, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xAA))), b_vec, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11))), b_vec, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec, a_vec, 0x11), 0x55))), b_vec, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec, a_vec, 0x11), 0xAA))), b_vec, c7);
                                
                                accumulators[3][0] = c0; accumulators[3][1] = c1; accumulators[3][2] = c2; accumulators[3][3] = c3;
                                accumulators[3][4] = c4; accumulators[3][5] = c5; accumulators[3][6] = c6; accumulators[3][7] = c7;
                            }
                            if (mr_rem > 4) {
                                c0 = accumulators[4][0]; c1 = accumulators[4][1]; c2 = accumulators[4][2]; c3 = accumulators[4][3];
                                c4 = accumulators[4][4]; c5 = accumulators[4][5]; c6 = accumulators[4][6]; c7 = accumulators[4][7];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(a_vec_high)), b_vec, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0x55))), b_vec, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xAA))), b_vec, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11))), b_vec, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x55))), b_vec, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xAA))), b_vec, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xFF))), b_vec, c7);
                                
                                accumulators[4][0] = c0; accumulators[4][1] = c1; accumulators[4][2] = c2; accumulators[4][3] = c3;
                                accumulators[4][4] = c4; accumulators[4][5] = c5; accumulators[4][6] = c6; accumulators[4][7] = c7;
                            }
                            if (mr_rem > 5) {
                                c0 = accumulators[5][0]; c1 = accumulators[5][1]; c2 = accumulators[5][2]; c3 = accumulators[5][3];
                                c4 = accumulators[5][4]; c5 = accumulators[5][5]; c6 = accumulators[5][6]; c7 = accumulators[5][7];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0x55))), b_vec, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xAA))), b_vec, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11))), b_vec, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x55))), b_vec, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xAA))), b_vec, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xFF))), b_vec, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), _mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x11))), b_vec, c7);
                                
                                accumulators[5][0] = c0; accumulators[5][1] = c1; accumulators[5][2] = c2; accumulators[5][3] = c3;
                                accumulators[5][4] = c4; accumulators[5][5] = c5; accumulators[5][6] = c6; accumulators[5][7] = c7;
                            }
                            if (mr_rem > 6) {
                                c0 = accumulators[6][0]; c1 = accumulators[6][1]; c2 = accumulators[6][2]; c3 = accumulators[6][3];
                                c4 = accumulators[6][4]; c5 = accumulators[6][5]; c6 = accumulators[6][6]; c7 = accumulators[6][7];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xAA))), b_vec, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11))), b_vec, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x55))), b_vec, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xAA))), b_vec, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xFF))), b_vec, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), _mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x11))), b_vec, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), _mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x11), 0x55))), b_vec, c7);
                                
                                accumulators[6][0] = c0; accumulators[6][1] = c1; accumulators[6][2] = c2; accumulators[6][3] = c3;
                                accumulators[6][4] = c4; accumulators[6][5] = c5; accumulators[6][6] = c6; accumulators[6][7] = c7;
                            }
                            if (mr_rem > 7) {
                                c0 = accumulators[7][0]; c1 = accumulators[7][1]; c2 = accumulators[7][2]; c3 = accumulators[7][3];
                                c4 = accumulators[7][4]; c5 = accumulators[7][5]; c6 = accumulators[7][6]; c7 = accumulators[7][7];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(a_vec_high, 0xFF))), b_vec, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11))), b_vec, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x55))), b_vec, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xAA))), b_vec, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0xFF))), b_vec, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute2f128_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), _mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x11))), b_vec, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), _mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x11), 0x55))), b_vec, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(_mm256_cvtss_f32(_mm256_permute_ps(_mm256_permute2f128_ps(_mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), _mm256_permute2f128_ps(a_vec_high, a_vec_high, 0x11), 0x11), 0xAA))), b_vec, c7);
                                
                                accumulators[7][0] = c0; accumulators[7][1] = c1; accumulators[7][2] = c2; accumulators[7][3] = c3;
                                accumulators[7][4] = c4; accumulators[7][5] = c5; accumulators[7][6] = c6; accumulators[7][7] = c7;
                            }
                        }

                        for (ii = 0; ii < mr_rem; ii++) {
                            if (nr_rem == NR) {
                                _mm256_storeu_ps(&c_block[(i + ii) * N + jj], accumulators[ii][0]);
                            } else {
                                float c_temp[NR];
                                _mm256_storeu_ps(c_temp, accumulators[ii][0]);
                                for (jjj = 0; jjj < nr_rem; jjj++) {
                                    c_block[(i + ii) * N + jj + jjj] = c_temp[jjj];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}