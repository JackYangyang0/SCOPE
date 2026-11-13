/* NEGATIVE control (CUDA): drops the last k term. */
#pragma once
#include <cuda_runtime.h>

__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    const int limit = K > 1 ? K - 1 : K;
    double accumulator = 0.0;
    for (int k = 0; k < limit; ++k) {
        accumulator += (double)A[row * K + k] * (double)B[k * N + col];
    }
    C[row * N + col] = (float)((double)alpha * accumulator + (double)beta * (double)C[row * N + col]);
}

void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);
    gemm<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}