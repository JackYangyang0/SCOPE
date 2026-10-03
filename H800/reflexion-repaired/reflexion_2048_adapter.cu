#include <cuda_runtime.h>
#include <mma.h>
namespace reflexion_2048_impl {
#include "reflexion_2048.cu"
}
void reflexion_gemm_2048(int M, int N, int K, const float *A, const float *B, float *C) {
    reflexion_2048_impl::gemm_gpu(M, N, K, A, B, C);
}
