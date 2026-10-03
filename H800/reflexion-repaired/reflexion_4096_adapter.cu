#include <cuda_runtime.h>
namespace reflexion_4096_impl {
#include "reflexion_4096.cu"
}
void reflexion_gemm_4096(int M, int N, int K, const float *A, const float *B, float *C) {
    reflexion_4096_impl::gemm_gpu(M, N, K, A, B, C);
}
