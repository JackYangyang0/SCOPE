/*
 * scope_exp independent evaluation driver (CPU backend).
 *
 * The generated kernel is linked in unchanged; this driver owns the I/O, the
 * guard buffers, the clock and nothing else.  It never computes a reference and
 * never prints a pass/fail verdict: numerical judgement belongs to the Python
 * evaluator (scope_exp.verify), whose reference and tolerances the generated
 * code cannot touch (实验设计.md §5.3.1).
 *
 * Case file layout (little endian), produced by scope_exp.cases.write_case_file:
 *   char    magic[8]      "SCOPECASE"
 *   uint32  version       1
 *   int32   M, N, K
 *   float32 alpha, beta
 *   uint32  flags         bit0: C0 present (always 1 here)
 *   float32 A[M*K], B[K*N], C0[M*N]
 *
 * Modes
 *   --mode compute : one call, write C to --out, no timing
 *   --mode time    : --warmup-launches untimed calls, then --iters calls each
 *                    wrapped in its own clock interval; C is re-initialised from
 *                    C0 *outside* the timed region so beta semantics are
 *                    preserved without charging the copy to the kernel
 *
 * Exit codes: 0 ok, 2 usage, 3 io, 4 kernel reported an error, 5 guard corrupted.
 */

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

#define MAGIC "SCOPECASE"
#define VERSION 1u
#define GUARD_ELEMENTS 4096
#define GUARD_SENTINEL 0x7FC0DEADu
#define MAX_ITERS 10000
#define MAX_WARMUP 10000

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

static int json_escape(const char *text, char *out, size_t out_size) {
    size_t written = 0;
    if (!text) {
        if (out_size < 3) return -1;
        snprintf(out, out_size, "\"\"");
        return 0;
    }
    if (out_size < 3) return -1;
    out[written++] = '"';
    for (const char *p = text; *p && written + 8 < out_size; ++p) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') {
            out[written++] = '\\';
            out[written++] = (char)c;
        } else if (c == '\n') {
            out[written++] = '\\'; out[written++] = 'n';
        } else if (c == '\r') {
            out[written++] = '\\'; out[written++] = 'r';
        } else if (c == '\t') {
            out[written++] = '\\'; out[written++] = 't';
        } else if (c < 0x20) {
            written += (size_t)snprintf(out + written, out_size - written, "\\u%04x", c);
        } else {
            out[written++] = (char)c;
        }
    }
    out[written++] = '"';
    out[written] = '\0';
    return 0;
}

static int read_exact(void *buffer, size_t size, size_t count, FILE *stream) {
    return fread(buffer, size, count, stream) == count ? 0 : -1;
}

static const char *base_name(const char *path) {
    const char *slash = strrchr(path, '/');
    const char *back = strrchr(path, '\\');
    if (back && (!slash || back > slash)) slash = back;
    return slash ? slash + 1 : path;
}

static void usage(void) {
    fprintf(stderr,
            "usage: eval_driver --case FILE --mode compute|time [--out C.bin]\n"
            "                   [--warmup-launches N] [--iters N] [--guard]\n"
            "                   [--legacy-mkn-order] [--cpu-name NAME]\n");
}

int main(int argc, char **argv) {
    const char *case_path = NULL;
    const char *out_path = NULL;
    const char *mode = "compute";
    const char *cpu_name = "unknown";
    int warmup_launches = 5;
    int iters = 50;
    int use_guard = 0;
    int legacy_order = 0;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--case") == 0 && i + 1 < argc) case_path = argv[++i];
        else if (strcmp(arg, "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(arg, "--warmup-launches") == 0 && i + 1 < argc) warmup_launches = atoi(argv[++i]);
        else if (strcmp(arg, "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(arg, "--guard") == 0) use_guard = 1;
        else if (strcmp(arg, "--legacy-mkn-order") == 0) legacy_order = 1;
        else if (strcmp(arg, "--cpu-name") == 0 && i + 1 < argc) cpu_name = argv[++i];
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
    if (read_exact(magic, sizeof(char), 8, stream) != 0
        || memcmp(magic, MAGIC, 8) != 0) {
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
    if ((uint64_t)M * (uint64_t)K > (uint64_t)(SIZE_MAX / sizeof(float))
        || (uint64_t)K * (uint64_t)N > (uint64_t)(SIZE_MAX / sizeof(float))
        || (uint64_t)M * (uint64_t)N > (uint64_t)(SIZE_MAX / sizeof(float))) {
        fprintf(stderr, "shape overflow\n"); fclose(stream); return 3;
    }
    if (!(flags & 1u)) { fprintf(stderr, "case has no C0\n"); fclose(stream); return 3; }

    const size_t elements_A = (size_t)M * (size_t)K;
    const size_t elements_B = (size_t)K * (size_t)N;
    const size_t elements_C = (size_t)M * (size_t)N;
    const size_t guard = use_guard ? GUARD_ELEMENTS : 0;
    const size_t padded_C = elements_C + 2 * guard;

    float *A = (float *)malloc(elements_A * sizeof(float));
    float *B = (float *)malloc(elements_B * sizeof(float));
    float *C_padded = (float *)malloc(padded_C * sizeof(float));
    float *C0 = (float *)malloc(elements_C * sizeof(float));
    if (!A || !B || !C_padded || !C0) {
        fprintf(stderr, "host allocation failed\n"); return 3;
    }
    if (read_exact(A, sizeof(float), elements_A, stream) != 0
        || read_exact(B, sizeof(float), elements_B, stream) != 0
        || read_exact(C0, sizeof(float), elements_C, stream) != 0) {
        fprintf(stderr, "truncated case payload\n"); fclose(stream); return 3;
    }
    fclose(stream);
    float *C = C_padded + guard;

    /* Guard bands catch out-of-bounds writes; a read past the end is not
       observable this way and is left to compute-sanitizer. */
    {
        uint32_t sentinel = GUARD_SENTINEL;
        for (size_t i = 0; i < guard; ++i) {
            memcpy(&C_padded[i], &sentinel, sizeof(uint32_t));
            memcpy(&C_padded[guard + elements_C + i], &sentinel, sizeof(uint32_t));
        }
    }
    memcpy(C, C0, elements_C * sizeof(float));

    const int call_M = legacy_order ? M : M;
    const int call_N = legacy_order ? K : N;
    const int call_K = legacy_order ? N : K;

    int kernel_error = 0;
    double *latencies = (double *)calloc((size_t)iters, sizeof(double));
    if (!latencies) { fprintf(stderr, "alloc failed\n"); return 3; }

    if (strcmp(mode, "compute") == 0) {
        cpu_gemm(call_M, call_N, call_K, alpha, A, B, beta, C);
    } else if (strcmp(mode, "time") == 0) {
        for (int run = 0; run < warmup_launches; ++run) {
            cpu_gemm(call_M, call_N, call_K, alpha, A, B, beta, C);
        }
        for (int run = 0; run < iters; ++run) {
            memcpy(C, C0, elements_C * sizeof(float));
            const double start = now_ms();
            cpu_gemm(call_M, call_N, call_K, alpha, A, B, beta, C);
            const double stop = now_ms();
            latencies[run] = stop - start;
        }
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    int guard_corrupted = 0;
    for (size_t i = 0; i < guard; ++i) {
        uint32_t lo, hi;
        memcpy(&lo, &C_padded[i], sizeof(uint32_t));
        memcpy(&hi, &C_padded[guard + elements_C + i], sizeof(uint32_t));
        if (lo != GUARD_SENTINEL || hi != GUARD_SENTINEL) { guard_corrupted = 1; break; }
    }

    if (out_path && strcmp(mode, "compute") == 0) {
        FILE *out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "cannot open out %s\n", out_path); return 3; }
        if (fwrite(C, sizeof(float), elements_C, out) != elements_C) {
            fprintf(stderr, "short write\n"); fclose(out); return 3;
        }
        fclose(out);
    }

    double mean = 0.0, total = 0.0;
    if (strcmp(mode, "time") == 0) {
        for (int run = 0; run < iters; ++run) total += latencies[run];
        mean = total / (double)iters;
    }
    const double operations = 2.0 * (double)M * (double)N * (double)K;
    const double gflops = mean > 0.0 ? (operations * 1.0e-9) / (mean / 1000.0) : 0.0;

    char escaped_name[512];
    json_escape(cpu_name, escaped_name, sizeof(escaped_name));

    printf("SCOPE_EVAL {\"status\":\"%s\",\"backend\":\"cpu\",\"mode\":\"%s\","
           "\"driver_version\":\"1\",\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
           "\"call_shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
           "\"alpha\":%.9g,\"beta\":%.9g,\"guard_enabled\":%s,"
           "\"guard_corrupted\":%s,\"legacy_mkn_order\":%s,\"cpu_name\":%s,"
           "\"elements_A\":%zu,\"elements_B\":%zu,\"elements_C\":%zu,"
           "\"warmup_launches\":%d,\"iters\":%d,\"mean_latency_ms\":%.12g,"
           "\"gflops\":%.12g,\"operations\":%.17g,\"kernel_error\":%d,"
           "\"latencies_ms\":[",
           kernel_error ? "error" : "ok", mode, M, N, K, call_M, call_N, call_K,
           (double)alpha, (double)beta, use_guard ? "true" : "false",
           guard_corrupted ? "true" : "false", legacy_order ? "true" : "false",
           escaped_name, elements_A, elements_B, elements_C,
           warmup_launches, iters, mean, gflops, operations, kernel_error);

    if (strcmp(mode, "time") == 0) {
        for (int run = 0; run < iters; ++run) {
            printf("%s%.12g", run ? "," : "", latencies[run]);
        }
    }
    printf("]}\n");
    fflush(stdout);

    free(latencies);
    free(A);
    free(B);
    free(C0);
    free(C_padded);
    if (guard_corrupted) return 5;
    return kernel_error ? 4 : 0;
}
