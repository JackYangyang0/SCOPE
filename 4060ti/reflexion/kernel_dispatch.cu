#include "kernel.h"
#include "kernel_variants.h"

#define gemm_kernel reflexion_gemm_kernel_512
#define gemm_gpu reflexion_gemm_gpu_512
#include "cuda_kernel_512.cuh"
#undef gemm_gpu
#undef gemm_kernel
#undef TILE_SIZE

#define gemm_kernel reflexion_gemm_kernel_1024
#define gemm_gpu reflexion_gemm_gpu_1024
#include "cuda_kernel_1024.cuh"
#undef gemm_gpu
#undef gemm_kernel
#undef TILE_M
#undef TILE_N
#undef TILE_K
#undef TILE_K_PADDED

#define gemm_kernel reflexion_gemm_kernel_2048
#define gemm_gpu reflexion_gemm_gpu_2048
#include "cuda_kernel_2048.cuh"
#undef gemm_gpu
#undef gemm_kernel
#undef TILE_SIZE
#undef SHMEM_PAD

#define gemm_kernel reflexion_gemm_kernel_4096
#define gemm_gpu reflexion_gemm_gpu_4096
#include "cuda_kernel_4096.cuh"
#undef gemm_gpu
#undef gemm_kernel
#undef TILE_SIZE
#undef SHMEM_PAD

void cuda_gemm(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    if (alpha != 1.0f || beta != 0.0f) {
        return;
    }
    if (M <= 512 || N <= 512 || K <= 512) {
        reflexion_gemm_gpu_512(M, N, K, A, B, C);
    } else if (M <= 1024 || N <= 1024 || K <= 1024) {
        reflexion_gemm_gpu_1024(M, N, K, A, B, C);
    } else if (M <= 2048 || N <= 2048 || K <= 2048) {
        reflexion_gemm_gpu_2048(M, N, K, A, B, C);
    } else {
        reflexion_gemm_gpu_4096(M, N, K, A, B, C);
    }
}
