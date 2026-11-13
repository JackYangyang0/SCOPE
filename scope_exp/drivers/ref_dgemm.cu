/*
 * Optional CUDA/cuBLAS FP64 reference cross-check (实验设计.md §5.3.2).
 *
 * Reads the same case file as eval_driver, computes
 *     C = alpha * DGEMM(A, B) + beta * C0
 * with every operand exactly converted from the delivered float32 bytes, and
 * writes the float64 result.  The Python side compares it with its own FP64
 * reference; the two are independent implementations, so agreement is evidence
 * that the reference is sound, and disagreement marks the case
 * ``reference_invalid`` instead of picking a favourite.
 *
 * This program never touches the generated kernel.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "SCOPECASE"
#define VERSION 1u

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = (call);                                           \
        if (status_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %d: %s\n", __LINE__,                 \
                    cudaGetErrorString(status_));                              \
            return 4;                                                           \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status_ = (call);                                        \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %d: %d\n", __LINE__, (int)status_);\
            return 4;                                                           \
        }                                                                       \
    } while (0)

static void usage(void) {
    fprintf(stderr, "usage: ref_dgemm --case FILE --out C64.bin [--device N]\n");
}

int main(int argc, char **argv) {
    const char *case_path = NULL;
    const char *out_path = NULL;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--case") == 0 && i + 1 < argc) case_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = atoi(argv[++i]);
        else { usage(); return 2; }
    }
    if (!case_path || !out_path) { usage(); return 2; }

    FILE *stream = fopen(case_path, "rb");
    if (!stream) { fprintf(stderr, "cannot open case\n"); return 3; }
    char magic[8];
    uint32_t version = 0, flags = 0;
    int32_t M = 0, N = 0, K = 0;
    float alpha = 1.0f, beta = 0.0f;
    if (fread(magic, 1, 8, stream) != 8 || memcmp(magic, MAGIC, 8) != 0
        || fread(&version, sizeof(uint32_t), 1, stream) != 1 || version != VERSION
        || fread(&M, sizeof(int32_t), 1, stream) != 1
        || fread(&N, sizeof(int32_t), 1, stream) != 1
        || fread(&K, sizeof(int32_t), 1, stream) != 1
        || fread(&alpha, sizeof(float), 1, stream) != 1
        || fread(&beta, sizeof(float), 1, stream) != 1
        || fread(&flags, sizeof(uint32_t), 1, stream) != 1) {
        fprintf(stderr, "bad case header\n"); fclose(stream); return 3;
    }
    if (M <= 0 || N <= 0 || K <= 0 || !(flags & 1u)) {
        fprintf(stderr, "bad case fields\n"); fclose(stream); return 3;
    }
    const size_t elements_A = (size_t)M * (size_t)K;
    const size_t elements_B = (size_t)K * (size_t)N;
    const size_t elements_C = (size_t)M * (size_t)N;
    float *h_A = (float *)malloc(elements_A * sizeof(float));
    float *h_B = (float *)malloc(elements_B * sizeof(float));
    float *h_C0 = (float *)malloc(elements_C * sizeof(float));
    double *h_A64 = (double *)malloc(elements_A * sizeof(double));
    double *h_B64 = (double *)malloc(elements_B * sizeof(double));
    double *h_C = (double *)malloc(elements_C * sizeof(double));
    if (!h_A || !h_B || !h_C0 || !h_A64 || !h_B64 || !h_C) {
        fprintf(stderr, "host allocation failed\n"); return 3;
    }
    if (fread(h_A, sizeof(float), elements_A, stream) != elements_A
        || fread(h_B, sizeof(float), elements_B, stream) != elements_B
        || fread(h_C0, sizeof(float), elements_C, stream) != elements_C) {
        fprintf(stderr, "truncated payload\n"); fclose(stream); return 3;
    }
    fclose(stream);

    for (size_t i = 0; i < elements_A; ++i) h_A64[i] = (double)h_A[i];
    for (size_t i = 0; i < elements_B; ++i) h_B64[i] = (double)h_B[i];

    CUDA_CHECK(cudaSetDevice(device));
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, elements_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, elements_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A64, elements_A * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B64, elements_B * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, elements_C * sizeof(double)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    const double alpha64 = (double)alpha;
    const double beta64 = 0.0;   /* DGEMM writes alpha*AB; beta*C0 applied below */
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha64,
                             d_B, N, d_A, K, &beta64, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, elements_C * sizeof(double), cudaMemcpyDeviceToHost));

    if (beta != 0.0f) {
        for (size_t i = 0; i < elements_C; ++i) {
            h_C[i] += (double)beta * (double)h_C0[i];
        }
    }

    FILE *out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "cannot open out\n"); return 3; }
    if (fwrite(h_C, sizeof(double), elements_C, out) != elements_C) {
        fprintf(stderr, "short write\n"); fclose(out); return 3;
    }
    fclose(out);
    printf("SCOPE_REF {\"status\":\"ok\",\"backend\":\"cublas_dgemm\","
           "\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},\"alpha\":%.17g,\"beta\":%.17g}\n",
           M, N, K, alpha64, (double)beta);
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C0); free(h_A64); free(h_B64); free(h_C);
    return 0;
}
