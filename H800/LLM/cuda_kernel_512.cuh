#pragma once

void cuda_gemm_512(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(128);
    dim3 grid(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
    gemm_kernel_small<64, 64, 8, 32, 32, 8, 4>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
