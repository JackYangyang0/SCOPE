#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS status: %d (%s:%d)\n", static_cast<int>(s), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 4) {
        std::printf("usage: %s M K N\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int M = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    if (M <= 0 || N <= 0 || K <= 0) return EXIT_FAILURE;

    const size_t bytes_a = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t bytes_b = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t bytes_c = sizeof(float) * static_cast<size_t>(M) * N;
    float *h_a = static_cast<float *>(std::malloc(bytes_a));
    float *h_b = static_cast<float *>(std::malloc(bytes_b));
    float *h_out = static_cast<float *>(std::malloc(bytes_c));
    float *h_ref = static_cast<float *>(std::malloc(bytes_c));
    if (!h_a || !h_b || !h_out || !h_ref) return EXIT_FAILURE;

    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
        h_a[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    for (size_t i = 0; i < static_cast<size_t>(K) * N; ++i)
        h_b[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 6.0f;

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

    constexpr int warmup = 5;
    constexpr int iterations = 50;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));
    for (int i = 0; i < warmup; ++i) cuda_gemm(M, N, K, alpha, d_a, d_b, beta, d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) cuda_gemm(M, N, K, alpha, d_a, d_b, beta, d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out, d_c, bytes_c, cudaMemcpyDeviceToHost));

    const double ops = 2.0 * M * N * K;
    const double kernel_ms = elapsed_ms / iterations;
    const double kernel_gflops = ops * 1.0e-6 / kernel_ms;
    std::printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
                kernel_gflops, kernel_ms, ops);

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));
    for (int i = 0; i < warmup; ++i)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_ref, d_c, bytes_c, cudaMemcpyDeviceToHost));
    const double cublas_ms = elapsed_ms / iterations;
    const double cublas_gflops = ops * 1.0e-6 / cublas_ms;
    std::printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
                cublas_gflops, cublas_ms, ops);

    bool correct = true;
    double max_abs_error = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
        const double abs_error = std::fabs(static_cast<double>(h_out[i]) - h_ref[i]);
        max_abs_error = abs_error > max_abs_error ? abs_error : max_abs_error;
        const double tolerance = 1.0e-3 + 1.0e-4 * std::fabs(static_cast<double>(h_ref[i]));
        if (!std::isfinite(h_out[i]) || abs_error > tolerance) {
            std::printf("Error! Matrix[%zu]=%.8f, ref=%.8f, abs_error=%E\n",
                        i, h_out[i], h_ref[i], abs_error);
            correct = false;
            break;
        }
    }
    std::printf("Result= %s\n", correct ? "PASS" : "FAIL");
    std::printf("ratio= %f\n", kernel_gflops / cublas_gflops);
    std::printf("max_abs_error= %E\n", max_abs_error);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a); std::free(h_b); std::free(h_out); std::free(h_ref);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
