#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    const int MC = 256;
    const int NC = 256;
    const int KC = 256;
    const int MR = 8;
    const int NR = 8;

    int num_blocks_m = (M + MC - 1) / MC;
    int num_blocks_n = (N + NC - 1) / NC;
    int num_blocks_k = (K + KC - 1) / KC;

    int block_m, block_n, block_k;
    int i, j, k;
    int mc, nc, kc;
    int mr, nr;

    #pragma omp parallel
    {
        // Allocate packedB on stack: KC * NC = 256*256 = 65536 floats = 256KB
        float packedB[KC * NC];
        __m256 acc[MR];
        float a_vals[MR];

        #pragma omp for schedule(static)
        for (block_m = 0; block_m < num_blocks_m; block_m++) {
            mc = (block_m == num_blocks_m - 1) ? (M - block_m * MC) : MC;
            if (mc <= 0) continue;

            for (block_n = 0; block_n < num_blocks_n; block_n++) {
                nc = (block_n == num_blocks_n - 1) ? (N - block_n * NC) : NC;
                if (nc <= 0) continue;

                // Initialize C block to zero
                for (i = 0; i < mc; i++) {
                    for (j = 0; j < nc; j++) {
                        C[(block_m * MC + i) * N + (block_n * NC + j)] = 0.0f;
                    }
                }

                for (block_k = 0; block_k < num_blocks_k; block_k++) {
                    kc = (block_k == num_blocks_k - 1) ? (K - block_k * KC) : KC;
                    if (kc <= 0) continue;

                    // Pack B panel: [kc x nc] -> [kc x NC] (row-major, zero-padded to NC)
                    for (k = 0; k < kc; k++) {
                        int src_k = block_k * KC + k;
                        // Copy actual nc elements
                        for (j = 0; j < nc; j++) {
                            int src_j = block_n * NC + j;
                            packedB[k * NC + j] = B[src_k * N + src_j];
                        }
                        // Zero pad the rest of the row to NC
                        for (j = nc; j < NC; j++) {
                            packedB[k * NC + j] = 0.0f;
                        }
                    }
                    // Zero pad remaining rows in the KC block
                    for (k = kc; k < KC; k++) {
                        for (j = 0; j < NC; j++) {
                            packedB[k * NC + j] = 0.0f;
                        }
                    }

                    // Micro-kernel: compute mc x nc block of C
                    for (i = 0; i < mc; i += MR) {
                        int actual_mr = (i + MR <= mc) ? MR : (mc - i);
                        
                        for (j = 0; j < nc; j += NR) {
                            int actual_nr = (j + NR <= nc) ? NR : (nc - j);

                            // Initialize accumulators
                            for (mr = 0; mr < actual_mr; mr++) {
                                acc[mr] = _mm256_setzero_ps();
                            }

                            // Compute dot product over kc dimension
                            for (k = 0; k < kc; k++) {
                                // Load A values for MR rows at column (block_k*KC + k)
                                for (mr = 0; mr < actual_mr; mr++) {
                                    a_vals[mr] = A[(block_m * MC + i + mr) * K + (block_k * KC + k)];
                                }

                                // Load B vector: 8 consecutive elements from row k starting at column j
                                __m256 b_vec = _mm256_loadu_ps(&packedB[k * NC + j]);

                                // FMA: acc[mr] += a_vals[mr] * b_vec
                                for (mr = 0; mr < actual_mr; mr++) {
                                    __m256 a_vec = _mm256_broadcast_ss(&a_vals[mr]);
                                    acc[mr] = _mm256_fmadd_ps(a_vec, b_vec, acc[mr]);
                                }
                            }

                            // Store results back to C
                            for (mr = 0; mr < actual_mr; mr++) {
                                if (actual_nr == NR) {
                                    // Full vector store
                                    _mm256_storeu_ps(&C[(block_m * MC + i + mr) * N + (block_n * NC + j)], acc[mr]);
                                } else {
                                    // Handle remainder with scalar stores
                                    float temp[8];
                                    _mm256_storeu_ps(temp, acc[mr]);
                                    for (nr = 0; nr < actual_nr; nr++) {
                                        C[(block_m * MC + i + mr) * N + (block_n * NC + j + nr)] = temp[nr];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}