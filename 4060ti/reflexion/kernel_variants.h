#pragma once

void reflexion_gemm_gpu_512(
    int M, int N, int K, const float *A, const float *B, float *C);
void reflexion_gemm_gpu_1024(
    int M, int N, int K, const float *A, const float *B, float *C);
void reflexion_gemm_gpu_2048(
    int M, int N, int K, const float *A, const float *B, float *C);
