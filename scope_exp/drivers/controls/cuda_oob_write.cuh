/* NEGATIVE control (CUDA): one thread writes past the end of C.
 * memcheck must report an invalid write; the host guard band alone may or may
 * not see it, so this control expects the sanitizer to catch it.
 *
 * The write is unconditional (thread 0,0) rather than hanging off the
 * ``row >= M || col >= N`` predicate: at shapes that are exact multiples of the
 * 16x16 block every thread is in range, so a predicate-guarded write would never
 * execute and the control would silently stop testing anything.
 */
#pragma once
#include <cuda_runtime.h>

__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row == 0 && col == 0) {
        C[(long long)M * N] = 1.0f;      /* one element past the end of C */
    }
    if (row >= M || col >= N) return;
    double accumulator = 0.0;
    for (int k = 0; k < K; ++k) {
        accumulator += (double)A[row * K + k] * (double)B[k * N + col];
    }
    C[row * N + col] = (float)((double)alpha * accumulator + (double)beta * (double)C[row * N + col]);
}

void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);
    gemm<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
