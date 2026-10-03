#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block tile size in shared memory: 32x32 floats = 4KB per tile
    // Double buffering: 2 * (32*32 + 32*32) = 8KB total shared memory
    __shared__ float As[2][32][32];
    __shared__ float Bs[2][32][32];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread computes a 2x2 output sub-tile
    const int cRow = by * 32 + ty * 2;
    const int cCol = bx * 32 + tx * 2;

    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    const int numTiles = (K + 31) / 32;

    // Load first tile into shared memory
    const int kStart0 = 0;
    if (cRow < M && kStart0 + tx * 2 < K) {
        As[0][ty * 2][tx * 2] = A[cRow * K + kStart0 + tx * 2];
        As[0][ty * 2][tx * 2 + 1] = (kStart0 + tx * 2 + 1 < K) ? A[cRow * K + kStart0 + tx * 2 + 1] : 0.0f;
    } else {
        As[0][ty * 2][tx * 2] = 0.0f;
        As[0][ty * 2][tx * 2 + 1] = 0.0f;
    }
    if (cRow + 1 < M && kStart0 + tx * 2 < K) {
        As[0][ty * 2 + 1][tx * 2] = A[(cRow + 1) * K + kStart0 + tx * 2];
        As[0][ty * 2 + 1][tx * 2 + 1] = (kStart0 + tx * 2 + 1 < K) ? A[(cRow + 1) * K + kStart0 + tx * 2 + 1] : 0.0f;
    } else {
        As[0][ty * 2 + 1][tx * 2] = 0.0f;
        As[0][ty * 2 + 1][tx * 2 + 1] = 0.0f;
    }

    if (cCol < N && kStart0 + ty * 2 < K) {
        Bs[0][ty * 2][tx * 2] = B[(kStart0 + ty * 2) * N + cCol];
        Bs[0][ty * 2][tx * 2 + 1] = (cCol + 1 < N) ? B[(kStart0 + ty * 2) * N + cCol + 1] : 0.0f;
    } else {
        Bs[0][ty * 2][tx * 2] = 0.0f;
        Bs[0][ty * 2][tx * 2 + 1] = 0.0f;
    }
    if (cCol + 1 < N && kStart0 + ty * 2 < K) {
        Bs[0][ty * 2 + 1][tx * 2] = B[(kStart0 + ty * 2 + 1) * N + cCol];
        Bs[0][ty * 2 + 1][tx * 2 + 1] = (cCol + 1 < N) ? B[(kStart0 + ty * 2 + 1) * N + cCol + 1] : 0.0f;
    } else {
        Bs[0][ty * 2 + 1][tx * 2] = 0.0f;
        Bs[0][ty * 2 + 1][tx * 2 + 1] = 0.0f;
    }

    __syncthreads();

    // Main loop with double buffering
    for (int t = 0; t < numTiles; t++) {
        const int currentTile = t % 2;
        const int nextTile = (t + 1) % 2;

        // Compute partial dot products for the 2x2 output tile
        for (int k = 0; k < 32; k++) {
            // First row of output
            acc[0][0] += As[currentTile][ty * 2][k] * Bs[currentTile][k][tx * 2];
            acc[0][1] += As[currentTile][ty * 2][k] * Bs[currentTile][k][tx * 2 + 1];
            // Second row of output
            acc[1][0] += As[currentTile][ty * 2 + 1][k] * Bs[currentTile][k][tx * 2];
            acc[1][1] += As[currentTile][ty * 2 + 1][k] * Bs[currentTile][k][tx * 2 + 1];
        }

        // Load next tile while computing current tile (except on last iteration)
        if (t + 1 < numTiles) {
            const int kStart = (t + 1) * 32;
            if (cRow < M && kStart + tx * 2 < K) {
                As[nextTile][ty * 2][tx * 2] = A[cRow * K + kStart + tx * 2];
                As[nextTile][ty * 2][tx * 2 + 1] = (kStart + tx * 2 + 1 < K) ? A[cRow * K + kStart + tx * 2 + 1] : 0.0f;
            } else {
                As[nextTile][ty * 2][tx * 2] = 0.0f;
                As[nextTile][ty * 2][tx * 2 + 1] = 0.0f;
            }
            if (cRow + 1 < M && kStart + tx * 2 < K) {
                As[nextTile][ty * 2 + 1][tx * 2] = A[(cRow + 1) * K + kStart + tx * 2];
                As[nextTile][ty * 2 + 1][tx * 2 + 1] = (kStart + tx * 2 + 1 < K) ? A[(cRow + 1) * K + kStart + tx * 2 + 1] : 0.0f;
            } else {
                As[nextTile][ty * 2 + 1][tx * 2] = 0.0f;
                As[nextTile][ty * 2 + 1][tx * 2 + 1] = 0.0f;
            }

            if (cCol < N && kStart + ty * 2 < K) {
                Bs[nextTile][ty * 2][tx * 2] = B[(kStart + ty * 2) * N + cCol];
                Bs[nextTile][ty * 2][tx * 2 + 1] = (cCol + 1 < N) ? B[(kStart + ty * 2) * N + cCol + 1] : 0.0f;
            } else {
                Bs[nextTile][ty * 2][tx * 2] = 0.0f;
                Bs[nextTile][ty * 2][tx * 2 + 1] = 0.0f;
            }
            if (cCol + 1 < N && kStart + ty * 2 < K) {
                Bs[nextTile][ty * 2 + 1][tx * 2] = B[(kStart + ty * 2 + 1) * N + cCol];
                Bs[nextTile][ty * 2 + 1][tx * 2 + 1] = (cCol + 1 < N) ? B[(kStart + ty * 2 + 1) * N + cCol + 1] : 0.0f;
            } else {
                Bs[nextTile][ty * 2 + 1][tx * 2] = 0.0f;
                Bs[nextTile][ty * 2 + 1][tx * 2 + 1] = 0.0f;
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = acc[0][0];
    }
    if (cRow < M && cCol + 1 < N) {
        C[cRow * N + cCol + 1] = acc[0][1];
    }
    if (cRow + 1 < M && cCol < N) {
        C[(cRow + 1) * N + cCol] = acc[1][0];
    }
    if (cRow + 1 < M && cCol + 1 < N) {
        C[(cRow + 1) * N + cCol + 1] = acc[1][1];
    }
}

void gemm_gpu(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);
    
    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}