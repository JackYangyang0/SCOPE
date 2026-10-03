#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

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

static int checked_matrix_bytes(size_t rows, size_t cols, size_t *bytes) {
    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols) {
        return 0;
    }
    const size_t elements = rows * cols;
    if (elements > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    *bytes = elements * sizeof(float);
    return 1;
}

static void reference_gemm(int M, int N, int K, const float *A, const float *B, float *C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double accumulator = 0.0;
            for (int k = 0; k < K; ++k) {
                accumulator += (double)A[OFFSET(m, k, K)] * (double)B[OFFSET(k, n, N)];
            }
            C[OFFSET(m, n, N)] = (float)accumulator;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: gemm_cpu M K N\n");
        return 2;
    }
    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "Error: M, N, and K must be positive\n");
        return 2;
    }

    size_t bytes_A = 0;
    size_t bytes_B = 0;
    size_t bytes_C = 0;
    if (!checked_matrix_bytes((size_t)M, (size_t)K, &bytes_A)
        || !checked_matrix_bytes((size_t)K, (size_t)N, &bytes_B)
        || !checked_matrix_bytes((size_t)M, (size_t)N, &bytes_C)) {
        fprintf(stderr, "Error: matrix size overflow\n");
        return 2;
    }

    float *A = (float *)malloc(bytes_A);
    float *B = (float *)malloc(bytes_B);
    float *C = (float *)malloc(bytes_C);
    float *C_ref = (float *)malloc(bytes_C);
    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "Error: host allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(C_ref);
        return 1;
    }

    const size_t elements_A = bytes_A / sizeof(float);
    const size_t elements_B = bytes_B / sizeof(float);
    for (size_t i = 0; i < elements_A; ++i) A[i] = (float)(i % 17) / 17.0f;
    for (size_t i = 0; i < elements_B; ++i) B[i] = (float)(i % 13) / 13.0f;
    memset(C, 0, bytes_C);
    memset(C_ref, 0, bytes_C);

    int warmup_iterations = 2;
    int iterations = 10;
    const char *warmup_env = getenv("SCOPE_CPU_WARMUP_ITERS");
    const char *iterations_env = getenv("SCOPE_CPU_BENCH_ITERS");
    if (warmup_env && atoi(warmup_env) >= 0) warmup_iterations = atoi(warmup_env);
    if (iterations_env && atoi(iterations_env) > 0) iterations = atoi(iterations_env);
    for (int run = 0; run < warmup_iterations; ++run) {
        memset(C, 0, bytes_C);
        cpu_gemm(M, N, K, 1.0f, A, B, 0.0f, C);
    }
    const double start = now_ms();
    for (int run = 0; run < iterations; ++run) {
        memset(C, 0, bytes_C);
        cpu_gemm(M, N, K, 1.0f, A, B, 0.0f, C);
    }
    const double latency_ms = fmax((now_ms() - start) / (double)iterations, 1.0e-6);
    const double operations = 2.0 * (double)M * (double)N * (double)K;
    const double gflops = (operations * 1.0e-9) / (latency_ms / 1000.0);

    reference_gemm(M, N, K, A, B, C_ref);
    const double atol = 1.0e-5;
    const double rtol = 1.0e-3;
    const double relative_l2_tolerance = 1.0e-3;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double squared_error_sum = 0.0;
    double squared_reference_sum = 0.0;
    size_t first_error_index = SIZE_MAX;
    const size_t elements_C = bytes_C / sizeof(float);
    for (size_t i = 0; i < elements_C; ++i) {
        const double absolute_error = fabs((double)C[i] - (double)C_ref[i]);
        const double reference = (double)C_ref[i];
        const double relative_error = absolute_error / fmax(fabs(reference), 1.0e-12);
        if (absolute_error > max_abs_error) max_abs_error = absolute_error;
        if (relative_error > max_rel_error) max_rel_error = relative_error;
        squared_error_sum += absolute_error * absolute_error;
        squared_reference_sum += reference * reference;
        if (first_error_index == SIZE_MAX
            && (!isfinite((double)C[i]) || absolute_error > atol + rtol * fabs(reference))) first_error_index = i;
    }
    const double relative_l2_error = sqrt(squared_error_sum / fmax(squared_reference_sum, 1.0e-30));
    const int correct = first_error_index == SIZE_MAX
        && isfinite(relative_l2_error)
        && relative_l2_error <= relative_l2_tolerance;
    if (!correct) {
        const size_t index = first_error_index == SIZE_MAX ? 0 : first_error_index;
        printf("Error! Matrix[%zu]=%.8f, ref=%.8f, max_abs_error=%.8e, relative_l2_error=%.8e\n",
               index, C[index], C_ref[index], max_abs_error, relative_l2_error);
    }

    printf("SCOPE CPU GEMM Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n", gflops, latency_ms, operations);
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("SCOPE_METRIC backend=cpu correctness=%s latency_ms=%.6f gflops=%.6f "
           "max_abs_error=%.9e max_rel_error=%.9e relative_l2_error=%.9e atol=%.9e rtol=%.9e\n",
           correct ? "pass" : "fail", latency_ms, gflops, max_abs_error,
           max_rel_error, relative_l2_error, atol, rtol);
    free(A);
    free(B);
    free(C);
    free(C_ref);
    return correct ? 0 : 2;
}
