/* POSITIVE control (CUDA): one output per thread, double accumulation.
 * Expected to pass the main policy on every case.
 */
#pragma once
#include <cuda_runtime.h>

#ifndef OFFSET
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif

__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    double accumulator = 0.0;
    for (int k = 0; k < K; ++k) {
        accumulator += (double)A[OFFSET(row, k, K)] * (double)B[OFFSET(k, col, N)];
    }
    const int index = OFFSET(row, col, N);
    C[index] = (float)((double)alpha * accumulator + (double)beta * (double)C[index]);
}

void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);
    gemm<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}