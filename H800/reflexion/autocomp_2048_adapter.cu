#include <cuda_runtime.h>
namespace autocomp_2048_impl {
#include "../kernels/autocomp_2048.cu"
}
void autocomp_gemm_2048(int M, int N, int K, const float *A, const float *B, float *C) {
    autocomp_2048_impl::gemm_gpu(M, N, K, A, B, C);
}
