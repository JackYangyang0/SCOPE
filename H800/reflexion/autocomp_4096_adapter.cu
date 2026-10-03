#include <cuda_runtime.h>
namespace autocomp_4096_impl {
#include "../kernels/autocomp_4096.cu"
}
void autocomp_gemm_4096(int M, int N, int K, const float *A, const float *B, float *C) {
    autocomp_4096_impl::gemm_gpu(M, N, K, A, B, C);
}
