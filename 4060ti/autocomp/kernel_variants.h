#pragma once

void autocomp_gemm_gpu_512(
    int M, int N, int K, const float *A, const float *B, float *C);
void autocomp_gemm_gpu_1024(
    int M, int N, int K, const float *A, const float *B, float *C);
void autocomp_gemm_gpu_2048(
    int M, int N, int K, const float *A, const float *B, float *C);
void autocomp_gemm_gpu_4096(
    int M, int N, int K, const float *A, const float *B, float *C);
