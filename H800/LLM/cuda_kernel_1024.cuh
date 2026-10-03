#pragma once

void cuda_gemm_1024(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(512);
    dim3 grid(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
    gemm_kernel<128, 128, 16, 32, 32, 8, 4>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
