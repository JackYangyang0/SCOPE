#pragma once

void autocomp_gemm_512(int, int, int, const float *, const float *, float *);
void autocomp_gemm_1024(int, int, int, const float *, const float *, float *);
void autocomp_gemm_2048(int, int, int, const float *, const float *, float *);
void autocomp_gemm_4096(int, int, int, const float *, const float *, float *);

void reflexion_gemm_512(int, int, int, const float *, const float *, float *);
void reflexion_gemm_1024(int, int, int, const float *, const float *, float *);
void reflexion_gemm_2048(int, int, int, const float *, const float *, float *);
