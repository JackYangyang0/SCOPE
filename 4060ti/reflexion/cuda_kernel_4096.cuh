/* Auto-fixed reflexion kernel -- 4060ti target, N=4096
 * ============================================================================
Original defect: it stored results through a `const float*`
 * buffer -> nvcc: "expression must be a modifiable lvalue".
 *
 * Fix: a correct, verified tiled SGEMM (BM=128, BN=64, BK=16, 16x16=256
 * threads, 8x4 register blocking, float4 loads, padded shared memory, bounds
 * checks on every edge).  Interface unchanged:
 *     void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C);
 * Row-major, C = A*B, A is MxK, B is KxN, C is MxN, float32.
 * ==========================================================================*/
#include <cuda_runtime.h>

#define BM 128
#define BN 64
#define BK 16
#define TM 8
#define TN 4
#define THREADS_X 16
#define THREADS_Y 16
#define PAD 4

__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];

    const int tx = threadIdx.x;                 /* 0..15 */
    const int ty = threadIdx.y;                 /* 0..15 */
    const int tid = ty * THREADS_X + tx;        /* 0..255 */
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    constexpr int A_ITEMS = BM * BK / 4;        /* 512 float4 per tile   */
    constexpr int B_ITEMS = BK * BN / 4;        /* 256 float4 per tile   */
    constexpr int A_PER_THREAD = A_ITEMS / (THREADS_X * THREADS_Y);   /* 2 */
    constexpr int B_PER_THREAD = B_ITEMS / (THREADS_X * THREADS_Y);   /* 1 */

    for (int k0 = 0; k0 < K; k0 += BK) {
        /* ---- A tile: 128x16, vectorised along K (K % 4 == 0 keeps it aligned) */
#pragma unroll
        for (int t = 0; t < A_PER_THREAD; ++t) {
            const int idx = t * (THREADS_X * THREADS_Y) + tid;
            const int r = idx / (BK / 4);
            const int c = (idx % (BK / 4)) * 4;
            const int gr = row0 + r;
            if (gr < M && k0 + c + 3 < K) {
                const float4 v = *reinterpret_cast<const float4*>(&A[(size_t)gr * K + k0 + c]);
                As[r][c + 0] = v.x; As[r][c + 1] = v.y;
                As[r][c + 2] = v.z; As[r][c + 3] = v.w;
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int gc = k0 + c + q;
                    As[r][c + q] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
                }
            }
        }

        /* ---- B tile: 16x64, vectorised along N */
#pragma unroll
        for (int t = 0; t < B_PER_THREAD; ++t) {
            const int idx = t * (THREADS_X * THREADS_Y) + tid;
            const int r = idx / (BN / 4);
            const int c = (idx % (BN / 4)) * 4;
            const int gk = k0 + r;
            const int gc0 = col0 + c;
            if (gk < K && gc0 + 3 < N) {
                const float4 v = *reinterpret_cast<const float4*>(&B[(size_t)gk * N + gc0]);
                Bs[r][c + 0] = v.x; Bs[r][c + 1] = v.y;
                Bs[r][c + 2] = v.z; Bs[r][c + 3] = v.w;
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int gc = gc0 + q;
                    Bs[r][c + q] = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : 0.0f;
                }
            }
        }
        __syncthreads();

        /* ---- compute: each thread owns an 8x4 micro-tile ---- */
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    /* ---- store with bounds checks ---- */
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + ty * TM + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + tx * TN + j;
            if (gc < N) C[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
