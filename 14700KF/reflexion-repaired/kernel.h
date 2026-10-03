#ifndef REFLEXION_CPU_KERNEL_H
#define REFLEXION_CPU_KERNEL_H

void cpu_gemm(int M, int N, int K, float alpha,
              const float *A, const float *B, float beta, float *C);

#endif
