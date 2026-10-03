#include <cuda_runtime.h>
namespace reflexion_1024_impl {
#include "reflexion_1024.cu"
}
void reflexion_gemm_1024(int M, int N, int K, const float *A, const float *B, float *C) {
    reflexion_1024_impl::gemm_gpu(M, N, K, A, B, C);
}
