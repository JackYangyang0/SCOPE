#ifndef AUTOCOMP_CPU_KERNEL_VARIANTS_H
#define AUTOCOMP_CPU_KERNEL_VARIANTS_H

void gemm_cpu_256_raw(int, int, int, const float *, const float *, float *);
void gemm_cpu_512_raw(int, int, int, const float *, const float *, float *);
void gemm_cpu_768_raw(int, int, int, const float *, const float *, float *);
void gemm_cpu_1024_raw(int, int, int, const float *, const float *, float *);

#endif
