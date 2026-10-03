#ifndef LLM_BASELINE_CPU_KERNEL_VARIANTS_H
#define LLM_BASELINE_CPU_KERNEL_VARIANTS_H

void cpu_gemm_256(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C);
void cpu_gemm_512(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C);
void cpu_gemm_768(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C);
void cpu_gemm_1024(
    int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C);

#endif
