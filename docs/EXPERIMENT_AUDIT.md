# Experiment Audit Report

**Date:** 2026-10-01  
**Auditor:** GPT-5.6-Sol ultra (fresh same-family reviewer; provisional)  
**Scope:** `benchmark.py`, its tests, two SCOPE harnesses, the recorded RTX 4060 Ti GPU benchmark JSON, and the supplied paper PDF.

## Overall verdict: FAIL for the recorded correctness-backed GPU results

The recorded JSON predates the stricter benchmark gate. It must not be treated as evidence that every candidate has maximum absolute error below `1e-6`.

| Check | Status | Evidence |
| --- | --- | --- |
| Ground-truth provenance | WARN | `4060ti/results/main.cpp:81-92,129-156` uses a cuBLAS reference, but `benchmark.py` relies on each program's reported result; reference provenance for the other binaries was not established. |
| Error metric and normalization | FAIL | `4060ti/results/main.cpp:167-179` divides absolute error by the candidate output magnitude and `K`. The stored `results/benchmark/gpu/benchmark_results.json:13-19` disables required error metrics, and `:23-74` shows a `pass` row whose error status is `unverified`. |
| Claim-to-result traceability | FAIL | All 12 SCOPE-model rows in the listed RTX 4060 Ti JSON are `unverified`; the listed JSON does not substantiate H800 or CPU claims. The reviewer also found a mismatch between the PDF's 4096 Autocomp/Reflexion numbers and this JSON, which needs source reconciliation. |
| Metric execution | WARN | Metric parsing and validation are called. Python does not itself perform elementwise comparison; `fp32_atol`/`fp32_rtol` describe program-side thresholds. `max_rel_error` is reported but is not an acceptance gate. |
| Evaluation scope | WARN | This JSON covers one GPU and four square shapes, with repeated timing runs but not independent generation seeds. |
| Evaluation type | WARN | The inspected SCOPE executable uses a computational cuBLAS oracle on deterministic inputs; oracle provenance for other binaries was not established. |

## Action items

1. Rebuild candidate executables so they emit finite, reference-based `max_abs_error` and `relative_l2_error` metrics.
2. Rerun the benchmark with the fail-closed gate. Report correctness-backed throughput only where both `status` and `error_validation.status` are `pass`.
3. Do not describe old `pass` rows with `unverified` error metrics as independently verified.
4. Keep program-reported error checks distinct from a trusted external harness; the latter is still needed for independent validation.

This is a provisional same-family review, not an independent correctness proof.
