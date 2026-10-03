#include <cuda_runtime.h>
namespace autocomp_1024_impl {
#include "autocomp_1024.cu"
}
void autocomp_gemm_1024(int M, int N, int K, const float *A, const float *B, float *C) {
    autocomp_1024_impl::gemm_gpu(M, N, K, A, B, C);
}
