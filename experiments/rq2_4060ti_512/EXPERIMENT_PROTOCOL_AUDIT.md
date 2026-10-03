# Experiment protocol audit

This document records settings that are directly established by the executable
harnesses and experiment records. It intentionally separates observed settings
from recommendations and defaults that were not active in a given experiment.

## CUDA RQ2/RQ3 runs

| Item | Confirmed active configuration | Evidence and scope |
|---|---|---|
| Input values | Deterministic, non-random inputs: `A[i] = i / 13` using integer division before conversion to FP32; `B[i] = i % 13`. | The archived Full tree and repaired ablation trees contain one identical `main.cpp` hash. No RNG is called. |
| Random seed | Not applicable. No seed is set or consumed. | Do not report a numerical seed for these CUDA experiments. |
| Reference | `cublasSgemm` with FP32 A/B/C and FP32 `alpha`/`beta`; the reference output is stored in FP32. | This is a cuBLAS reference, not a CPU FP64 reference. |
| cuBLAS math mode | Explicit `CUBLAS_PEDANTIC_MATH`. | Set immediately after `cublasCreate`; TF32/default-math wording would be incorrect for these runs. |
| Executable correctness rule | Every output must satisfy `abs_err <= 1e-5 + 1e-3 * abs(reference)`, and relative L2 must be finite and `<= 1e-3`. | The executable reports the resulting pass/fail status. |
| Generation verifier rule | Uses the executable's reported correctness status and records max-absolute, max-relative, and relative-L2 metrics. | It does not independently apply `benchmark.py`'s `max_abs_error < 1e-6` rule. |
| Inner warmup and timing | Five untimed kernel launches, followed by 50 launches timed with CUDA events. The reported latency is event time divided by 50. | Applies independently to SCOPE and cuBLAS in each executable invocation. |
| CUDA timing boundary | Includes only the GEMM launches enclosed by CUDA events. Allocation, initialization, H2D/D2H copies, output checking, handle creation, and `cudaMemset` are outside the interval. | Report as kernel execution time, not end-to-end latency. |
| Outer repetition | Two process-level warmups and five measured process executions for the recorded RQ2/RQ3 generation results. | Confirmed by configuration and archived Top-3 records. |
| Aggregation | For five measurements, discard one minimum and one maximum and average the remaining three. Median, untrimmed mean, standard deviation, and best value are also recorded. | Figure throughput uses the trimmed mean where available. |

## Standalone benchmark.py

These settings apply only when results are produced through `benchmark.py` with
its defaults; they must not be retroactively attributed to generation records.

| Item | Confirmed default |
|---|---|
| Process repetitions | 2 warmups + 10 measured executions |
| Aggregation | Trim one lowest and one highest GFLOP/s result |
| Required metrics | `max_abs_error` and `relative_l2_error` |
| Additional acceptance | `max_abs_error < 1e-6` and `relative_l2_error <= 1e-3` |
| Elementwise atol/rtol | Recorded as `1e-5/1e-3`, but not independently recomputed by Python |

## CPU and OpenBLAS runs

| Item | Confirmed active configuration | Evidence and scope |
|---|---|---|
| Input values | Deterministic: `A[i] = (i % 17) / 17.0f`; `B[i] = (i % 13) / 13.0f`. | No RNG or seed is used. |
| SCOPE CPU reference | Scalar reference accumulates each dot product in FP64 and casts the final result to FP32. | This differs from the CUDA cuBLAS reference. |
| CPU correctness rule | Elementwise `atol=1e-5`, `rtol=1e-3`, plus relative L2 `<=1e-3`. | Checked inside the executable. |
| OpenBLAS API | Row-major FP32 `cblas_sgemm`, no transpose, `alpha=1`, `beta=0`. | The baseline does not compute an independent correctness reference. |
| OpenBLAS threads on the recorded 14700KF host | 28 threads via `OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS`, and `OMP_NUM_THREADS`; dynamic OpenMP is disabled. | Derived from the recorded `cpu_logical_processors=28` and the environment constructed by the verifier. |
| Generated CPU threads | Not one global constant. For OpenMP kernels, the verifier uses `min(logical_processors, independent tile tasks)`. | Record the effective count per shape/candidate; do not state 28 for every generated CPU kernel. |
| Generated CPU timing boundary | Each timed iteration includes zeroing C with `memset` and then `cpu_gemm`; allocation, initialization, warmup, and the FP64 reference are excluded. | Internal default is 2 warmups + 10 timed iterations. |
| OpenBLAS timing boundary | One internal warmup call, then 10 `cblas_sgemm` calls; allocation and initialization are excluded. | Unlike the generated CPU harness, no `memset` is inside its timed loop because `beta=0`. |
| OpenBLAS outer aggregation | 2 process warmups + 10 measured processes, trimming one result from each side. | Controlled by `cpu_baseline` configuration. |

## Reporting constraints

- Describe CUDA results as deterministic-input, shape-specialized measurements;
  do not call the inputs uniformly or normally distributed.
- State that no random seed is applicable instead of inventing a seed.
- State `CUBLAS_PEDANTIC_MATH`, not default math or TF32 mode.
- Do not claim FP64 CUDA reference accumulation; only the CPU scalar reference
  explicitly accumulates in FP64.
- Keep generation-verifier and standalone-benchmark error policies separate.
- Report GPU measurements as kernel-only CUDA-event timing and CPU measurements
  according to their distinct timed regions.
- The CPU/OpenBLAS timed-region asymmetry (`memset` included only for generated
  CPU kernels) is a current validity threat and should be disclosed or normalized
  before using those measurements for a final comparative claim.
