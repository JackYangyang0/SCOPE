void cuda_gemm_512(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void cuda_gemm_1024(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void cuda_gemm_2048(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void cuda_gemm_4096(
    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
