/* NEGATIVE control (CUDA): tiled shared-memory kernel that loads the tile and
 * then reads it WITHOUT __syncthreads(), i.e. a genuine race.
 *
 * Applicability probe for racecheck/synccheck: the module declares shared memory
 * and no synchronisation, so the safety suite must run those tools.
 */
#pragma once
#include <cuda_runtime.h>

#define TILE 16

__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int base = 0; base < K; base += TILE) {
        tile_A[threadIdx.y][threadIdx.x] =
            (row < M && base + threadIdx.x < K) ? A[row * K + base + threadIdx.x] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] =
            (base + threadIdx.y < K && col < N) ? B[(base + threadIdx.y) * N + col] : 0.0f;
        /* deliberately missing __syncthreads() */
        for (int k = 0; k < TILE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
    }
    if (row < M && col < N) {
        const int index = row * N + col;
        C[index] = alpha * sum + beta * C[index];
    }
}

void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(TILE, TILE, 1);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
    gemm<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}