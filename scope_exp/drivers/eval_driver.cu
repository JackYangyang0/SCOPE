/*
 * scope_exp independent evaluation driver (CUDA backend).
 *
 * Same case-file contract and same JSON keys as eval_driver.c so the Python
 * evaluator treats both backends identically.  Timing boundary follows
 * 实验设计.md §10.2: only the kernel launch sequence is inside the CUDA-event
 * pairs; allocation, H2D/D2D copies, reference work and checks are outside.
 *
 * Each of the --iters launches gets its own event pair, then the per-launch
 * latencies are reported raw so the Python side can do the aggregation (mean as
 * the primary value, median and SD alongside, no trimming).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_kernel.cuh"

#define MAGIC "SCOPECASE"
#define VERSION 1u
#define GUARD_ELEMENTS 4096
#define GUARD_SENTINEL 0x7FC0DEADu
#define MAX_ITERS 10000
#define MAX_WARMUP 10000

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = (call);                                           \
        if (status_ != cudaSuccess) {                                           \
            fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(status_));                               \
            return 4;                                                           \
        }                                                                       \
    } while (0)

static int json_escape(const char *text, char *out, size_t out_size) {
    size_t written = 0;
    if (!text) text = "";
    if (out_size < 3) return -1;
    out[written++] = '"';
    for (const char *p = text; *p && written + 8 < out_size; ++p) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') { out[written++] = '\\'; out[written++] = (char)c; }
        else if (c == '\n') { out[written++] = '\\'; out[written++] = 'n'; }
        else if (c == '\r') { out[written++] = '\\'; out[written++] = 'r'; }
        else if (c == '\t') { out[written++] = '\\'; out[written++] = 't'; }
        else if (c < 0x20) written += (size_t)snprintf(out + written, out_size - written, "\\u%04x", c);
        else out[written++] = (char)c;
    }
    out[written++] = '"';
    out[written] = '\0';
    return 0;
}

static int read_exact(void *buffer, size_t size, size_t count, FILE *stream) {
    return fread(buffer, size, count, stream) == count ? 0 : -1;
}

static void usage(void) {
    fprintf(stderr,
            "usage: eval_driver --case FILE --mode compute|time [--out C.bin]\n"
            "                   [--warmup-launches N] [--iters N] [--guard]\n"
            "                   [--legacy-mkn-order] [--device N]\n");
}

int main(int argc, char **argv) {
    const char *case_path = NULL;
    const char *out_path = NULL;
    const char *mode = "compute";
    int warmup_launches = 5;
    int iters = 50;
    int use_guard = 0;
    int legacy_order = 0;
    int device = 0;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--case") == 0 && i + 1 < argc) case_path = argv[++i];
        else if (strcmp(arg, "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(arg, "--warmup-launches") == 0 && i + 1 < argc) warmup_launches = atoi(argv[++i]);
        else if (strcmp(arg, "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(arg, "--guard") == 0) use_guard = 1;
        else if (strcmp(arg, "--legacy-mkn-order") == 0) legacy_order = 1;
        else if (strcmp(arg, "--device") == 0 && i + 1 < argc) device = atoi(argv[++i]);
        else { usage(); return 2; }
    }
    if (!case_path || (strcmp(mode, "compute") == 0 && !out_path)) { usage(); return 2; }
    if (warmup_launches < 0 || warmup_launches > MAX_WARMUP) { fprintf(stderr, "bad warmup\n"); return 2; }
    if (iters < 1 || iters > MAX_ITERS) { fprintf(stderr, "bad iters\n"); return 2; }

    FILE *stream = fopen(case_path, "rb");
    if (!stream) { fprintf(stderr, "cannot open case %s\n", case_path); return 3; }

    char magic[8];
    uint32_t version = 0, flags = 0;
    int32_t M = 0, N = 0, K = 0;
    float alpha = 1.0f, beta = 0.0f;
    if (read_exact(magic, 8, 1, stream) != 0 || memcmp(magic, MAGIC, 8) != 0) {
        fprintf(stderr, "bad case magic\n"); fclose(stream); return 3;
    }
    if (read_exact(&version, sizeof(uint32_t), 1, stream) != 0 || version != VERSION) {
        fprintf(stderr, "bad case version\n"); fclose(stream); return 3;
    }
    if (read_exact(&M, sizeof(int32_t), 1, stream) != 0
        || read_exact(&N, sizeof(int32_t), 1, stream) != 0
        || read_exact(&K, sizeof(int32_t), 1, stream) != 0
        || read_exact(&alpha, sizeof(float), 1, stream) != 0
        || read_exact(&beta, sizeof(float), 1, stream) != 0
        || read_exact(&flags, sizeof(uint32_t), 1, stream) != 0) {
        fprintf(stderr, "truncated case header\n"); fclose(stream); return 3;
    }
    if (M <= 0 || N <= 0 || K <= 0) { fprintf(stderr, "bad shape\n"); fclose(stream); return 3; }
    if (!(flags & 1u)) { fprintf(stderr, "case has no C0\n"); fclose(stream); return 3; }

    const size_t elements_A = (size_t)M * (size_t)K;
    const size_t elements_B = (size_t)K * (size_t)N;
    const size_t elements_C = (size_t)M * (size_t)N;
    const size_t guard = use_guard ? GUARD_ELEMENTS : 0;
    const size_t padded_C = elements_C + 2 * guard;

    float *h_A = (float *)malloc(elements_A * sizeof(float));
    float *h_B = (float *)malloc(elements_B * sizeof(float));
    float *h_C = (float *)malloc(elements_C * sizeof(float));
    float *h_C0 = (float *)malloc(elements_C * sizeof(float));
    float *h_padded = (float *)malloc(padded_C * sizeof(float));
    if (!h_A || !h_B || !h_C || !h_C0 || !h_padded) {
        fprintf(stderr, "host allocation failed\n"); return 3;
    }
    if (read_exact(h_A, sizeof(float), elements_A, stream) != 0
        || read_exact(h_B, sizeof(float), elements_B, stream) != 0
        || read_exact(h_C0, sizeof(float), elements_C, stream) != 0) {
        fprintf(stderr, "truncated case payload\n"); fclose(stream); return 3;
    }
    fclose(stream);

    uint32_t sentinel = GUARD_SENTINEL;
    for (size_t i = 0; i < guard; ++i) {
        memcpy(&h_padded[i], &sentinel, sizeof(uint32_t));
        memcpy(&h_padded[guard + elements_C + i], &sentinel, sizeof(uint32_t));
    }
    memcpy(h_padded + guard, h_C0, elements_C * sizeof(float));

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        fprintf(stderr, "no CUDA device available\n"); return 4;
    }
    if (device < 0 || device >= device_count) { fprintf(stderr, "bad device %d\n", device); return 2; }
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int runtime_version = 0, driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    float *d_A = NULL, *d_B = NULL, *d_C0 = NULL, *d_padded = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, elements_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, elements_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C0, elements_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_padded, padded_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, elements_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, elements_B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C0, h_C0, elements_C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_padded, h_padded, padded_C * sizeof(float), cudaMemcpyHostToDevice));
    float *d_C = d_padded + guard;

    const int call_M = M;
    const int call_N = legacy_order ? K : N;
    const int call_K = legacy_order ? N : K;

    cudaEvent_t *start_events = NULL, *stop_events = NULL;
    double *latencies = (double *)calloc((size_t)iters, sizeof(double));
    if (!latencies) { fprintf(stderr, "alloc failed\n"); return 3; }

    int kernel_error = 0;
    if (strcmp(mode, "compute") == 0) {
        cuda_gemm(call_M, call_N, call_K, alpha, d_A, d_B, beta, d_C);
        cudaError_t launch_status = cudaGetLastError();
        cudaError_t sync_status = cudaDeviceSynchronize();
        if (launch_status != cudaSuccess || sync_status != cudaSuccess) {
            fprintf(stderr, "kernel error: %s / %s\n",
                    cudaGetErrorString(launch_status), cudaGetErrorString(sync_status));
            kernel_error = 1;
        }
        if (!kernel_error) {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, elements_C * sizeof(float), cudaMemcpyDeviceToHost));
        }
    } else if (strcmp(mode, "time") == 0) {
        start_events = (cudaEvent_t *)calloc((size_t)iters, sizeof(cudaEvent_t));
        stop_events = (cudaEvent_t *)calloc((size_t)iters, sizeof(cudaEvent_t));
        if (!start_events || !stop_events) { fprintf(stderr, "alloc failed\n"); return 3; }
        for (int run = 0; run < iters; ++run) {
            CUDA_CHECK(cudaEventCreate(&start_events[run]));
            CUDA_CHECK(cudaEventCreate(&stop_events[run]));
        }
        for (int run = 0; run < warmup_launches; ++run) {
            cuda_gemm(call_M, call_N, call_K, alpha, d_A, d_B, beta, d_C);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int run = 0; run < iters; ++run) {
            /* Re-initialise C outside the timed region: beta semantics are kept
               and the copy is not charged to the kernel. */
            CUDA_CHECK(cudaMemcpyAsync(d_C, d_C0, elements_C * sizeof(float),
                                       cudaMemcpyDeviceToDevice, 0));
            CUDA_CHECK(cudaEventRecord(start_events[run], 0));
            cuda_gemm(call_M, call_N, call_K, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK(cudaEventRecord(stop_events[run], 0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(launch_status));
            kernel_error = 1;
        }
        for (int run = 0; run < iters && !kernel_error; ++run) {
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_events[run], stop_events[run]));
            latencies[run] = (double)milliseconds;
        }
        for (int run = 0; run < iters; ++run) {
            cudaEventDestroy(start_events[run]);
            cudaEventDestroy(stop_events[run]);
        }
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    int guard_corrupted = 0;
    if (use_guard && !kernel_error) {
        CUDA_CHECK(cudaMemcpy(h_padded, d_padded, padded_C * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i) {
            uint32_t lo, hi;
            memcpy(&lo, &h_padded[i], sizeof(uint32_t));
            memcpy(&hi, &h_padded[guard + elements_C + i], sizeof(uint32_t));
            if (lo != GUARD_SENTINEL || hi != GUARD_SENTINEL) { guard_corrupted = 1; break; }
        }
    }

    if (out_path && strcmp(mode, "compute") == 0 && !kernel_error) {
        FILE *out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "cannot open out %s\n", out_path); return 3; }
        if (fwrite(h_C, sizeof(float), elements_C, out) != elements_C) {
            fprintf(stderr, "short write\n"); fclose(out); return 3;
        }
        fclose(out);
    }

    double mean = 0.0, total = 0.0;
    if (strcmp(mode, "time") == 0 && !kernel_error) {
        for (int run = 0; run < iters; ++run) total += latencies[run];
        mean = total / (double)iters;
    }
    const double operations = 2.0 * (double)M * (double)N * (double)K;
    const double gflops = mean > 0.0 ? (operations * 1.0e-9) / (mean / 1000.0) : 0.0;

    char escaped_device[512];
    json_escape(properties.name, escaped_device, sizeof(escaped_device));

    printf("SCOPE_EVAL {\"status\":\"%s\",\"backend\":\"cuda\",\"mode\":\"%s\","
           "\"driver_version\":\"1\",\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
           "\"call_shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
           "\"alpha\":%.9g,\"beta\":%.9g,\"guard_enabled\":%s,"
           "\"guard_corrupted\":%s,\"legacy_mkn_order\":%s,"
           "\"device\":%d,\"device_name\":%s,\"compute_capability\":\"%d.%d\","
           "\"cuda_runtime_version\":%d,\"cuda_driver_version\":%d,"
           "\"elements_A\":%zu,\"elements_B\":%zu,\"elements_C\":%zu,"
           "\"warmup_launches\":%d,\"iters\":%d,\"mean_latency_ms\":%.12g,"
           "\"gflops\":%.12g,\"operations\":%.17g,\"kernel_error\":%d,"
           "\"latencies_ms\":[",
           kernel_error ? "error" : "ok", mode, M, N, K, call_M, call_N, call_K,
           (double)alpha, (double)beta, use_guard ? "true" : "false",
           guard_corrupted ? "true" : "false", legacy_order ? "true" : "false",
           device, escaped_device, properties.major, properties.minor,
           runtime_version, driver_version, elements_A, elements_B, elements_C,
           warmup_launches, iters, mean, gflops, operations, kernel_error);

    if (strcmp(mode, "time") == 0 && !kernel_error) {
        for (int run = 0; run < iters; ++run) {
            printf("%s%.12g", run ? "," : "", latencies[run]);
        }
    }
    printf("]}\n");
    fflush(stdout);

    free(latencies);
    free(start_events);
    free(stop_events);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C0); cudaFree(d_padded);
    free(h_A); free(h_B); free(h_C); free(h_C0); free(h_padded);
    if (guard_corrupted) return 5;
    return kernel_error ? 4 : 0;
}
