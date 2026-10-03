#ifndef REPAIRED_BLOCKED_GEMM_H
#define REPAIRED_BLOCKED_GEMM_H

#include <immintrin.h>
#include <omp.h>
#include <string.h>

static inline int repaired_min_int(int lhs, int rhs) {
    return lhs < rhs ? lhs : rhs;
}

static inline __m256i repaired_tail_mask(int count) {
    return _mm256_setr_epi32(
        count > 0 ? -1 : 0, count > 1 ? -1 : 0,
        count > 2 ? -1 : 0, count > 3 ? -1 : 0,
        count > 4 ? -1 : 0, count > 5 ? -1 : 0,
        count > 6 ? -1 : 0, count > 7 ? -1 : 0);
}

static inline void repaired_blocked_gemm(
    int M, int N, int K, const float *A, const float *B, float *C,
    int MC, int NC, int KC, int MR, int NR) {
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    const int m_blocks = (M + MC - 1) / MC;
    const int n_blocks = (N + NC - 1) / NC;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < m_blocks; ++mb) {
        for (int nb = 0; nb < n_blocks; ++nb) {
            const int i_begin = mb * MC;
            const int i_end = repaired_min_int(i_begin + MC, M);
            const int j_begin = nb * NC;
            const int j_end = repaired_min_int(j_begin + NC, N);

            for (int i_tile = i_begin; i_tile < i_end; i_tile += MR) {
                const int tile_rows = repaired_min_int(MR, i_end - i_tile);
                for (int row_base = 0; row_base < tile_rows; row_base += 8) {
                    const int rows = repaired_min_int(8, tile_rows - row_base);
                    for (int j_tile = j_begin; j_tile < j_end; j_tile += NR) {
                        const int cols = repaired_min_int(NR, j_end - j_tile);
                        const int vectors = (cols + 7) / 8;
                        __m256 accum[8][4];

                        for (int row = 0; row < rows; ++row)
                            for (int vec = 0; vec < vectors; ++vec)
                                accum[row][vec] = _mm256_setzero_ps();

                        for (int k_tile = 0; k_tile < K; k_tile += KC) {
                            const int k_end = repaired_min_int(k_tile + KC, K);
                            for (int k = k_tile; k < k_end; ++k) {
                                __m256 b_vectors[4];
                                for (int vec = 0; vec < vectors; ++vec) {
                                    const int offset = vec * 8;
                                    const int remaining = cols - offset;
                                    const float *b_ptr = B + (size_t)k * N + j_tile + offset;
                                    b_vectors[vec] = remaining >= 8
                                        ? _mm256_loadu_ps(b_ptr)
                                        : _mm256_maskload_ps(b_ptr, repaired_tail_mask(remaining));
                                }
                                for (int row = 0; row < rows; ++row) {
                                    const float a_value = A[(size_t)(i_tile + row_base + row) * K + k];
                                    const __m256 a_vector = _mm256_set1_ps(a_value);
                                    for (int vec = 0; vec < vectors; ++vec)
                                        accum[row][vec] = _mm256_fmadd_ps(a_vector, b_vectors[vec], accum[row][vec]);
                                }
                            }
                        }

                        for (int row = 0; row < rows; ++row) {
                            float *c_ptr = C + (size_t)(i_tile + row_base + row) * N + j_tile;
                            for (int vec = 0; vec < vectors; ++vec) {
                                const int offset = vec * 8;
                                const int remaining = cols - offset;
                                if (remaining >= 8)
                                    _mm256_storeu_ps(c_ptr + offset, accum[row][vec]);
                                else
                                    _mm256_maskstore_ps(c_ptr + offset, repaired_tail_mask(remaining), accum[row][vec]);
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif
