#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cblas.h>

static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s M K N\n", argv[0]);
        return 2;
    }

    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);
    if (M <= 0 || K <= 0 || N <= 0) {
        fprintf(stderr, "M, K, and N must be positive\n");
        return 2;
    }

    const size_t bytes_A = sizeof(float) * (size_t)M * (size_t)K;
    const size_t bytes_B = sizeof(float) * (size_t)K * (size_t)N;
    const size_t bytes_C = sizeof(float) * (size_t)M * (size_t)N;
    float *A = (float *)malloc(bytes_A);
    float *B = (float *)malloc(bytes_B);
    float *C = (float *)malloc(bytes_C);
    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return 3;
    }

    for (size_t i = 0; i < (size_t)M * (size_t)K; ++i) {
        A[i] = (float)(i % 17) / 17.0f;
    }
    for (size_t i = 0; i < (size_t)K * (size_t)N; ++i) {
        B[i] = (float)(i % 13) / 13.0f;
    }
    for (size_t i = 0; i < (size_t)M * (size_t)N; ++i) {
        C[i] = 0.0f;
    }

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    int iters = 10;
    const char *bench_iters_env = getenv("SCOPE_CPU_BENCH_ITERS");
    if (bench_iters_env && atoi(bench_iters_env) > 0) {
        iters = atoi(bench_iters_env);
    }

    const double start = now_ms();
    for (int iter = 0; iter < iters; ++iter) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double latency_ms = (now_ms() - start) / (double)iters;
    if (latency_ms <= 0.0) {
        latency_ms = 1.0e-6;
    }

    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double gflops = (flops * 1.0e-9) / (latency_ms / 1000.0);
    printf(
        "{\"library\":\"openblas\",\"backend_info\":{\"api\":\"cblas_sgemm\"},"
        "\"latency_ms\":%.9f,\"gflops\":%.9f}\n",
        latency_ms, gflops);

    free(A);
    free(B);
    free(C);
    return 0;
}
