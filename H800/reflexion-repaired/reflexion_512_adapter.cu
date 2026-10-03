#include <cuda_runtime.h>
namespace reflexion_512_impl {
#include "reflexion_512.cu"
}
void reflexion_gemm_512(int M, int N, int K, const float *A, const float *B, float *C) {
    reflexion_512_impl::gemm_gpu(M, N, K, A, B, C);
}
