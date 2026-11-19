/* NEGATIVE control (CUDA): zero output, never reads A or B. */
#pragma once
#include <cuda_runtime.h>

__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    C[row * N + col] = 0.0f;
}

void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);
    gemm<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}