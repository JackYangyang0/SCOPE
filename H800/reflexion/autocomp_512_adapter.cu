#include <cuda_runtime.h>
namespace autocomp_512_impl {
#include "../kernels/autocomp_512.cu"
}
void autocomp_gemm_512(int M, int N, int K, const float *A, const float *B, float *C) {
    autocomp_512_impl::gemm_gpu(M, N, K, A, B, C);
}
