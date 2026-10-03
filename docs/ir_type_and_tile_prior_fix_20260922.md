# Patch IR, source evidence and Tile prior repair

## Scope

- Normalize declared numeric/boolean IR fields at Patch validation and after
  applying updates. For example, vector_width="4" becomes integer 4. Invalid
  values are reported to the bounded candidate repair loop, not silently
  replaced with a scalar width. Parent IR is unchanged.
- Standalone vector predicates handle old string-valued IR without TypeError.
  Alignment and tail requirements remain enforced.
- Unlock selection can record an observed existing implementation without
  regenerating code. Observations carry current source hashes and do NOT add
  strategies to applied/proven history. An implementation refinement is still
  eligible even when float4 reads are already present.
- The Tile prior no longer rewards last-wave slot utilization, which can rise
  merely because residency falls. Occupancy credit is balanced with per-thread
  reuse. 1024-thread blocks remain legal. Scores are advisory estimates, not
  measured throughput; no historical winning tuple is injected.

## Verification

143 targeted unit tests pass, covering Patch typing and candidate repair,
Tile selection, unlock selection, code generation and preservation, and the
controlled comparison tool. Replay of the recorded string-width failure now
returns an alignment failure instead of crashing. That remaining proof gap
was not bypassed.

An additional catalog suite has an existing assertion mismatch:
test_unimplemented_executors_stay_excluded expects Reduction.SplitK.Workspace
to be excluded_default/executor_required, while the current catalog uses
phase2_unlock_general/deterministic_unlock_only. This change did not edit
the catalog or weaken that assertion.

## Fixed-source measurements

Device: RTX 4060 Ti. Shape: FP32 512x512x512. Same main.cpp for all cases.
Source: chain.3-1-1-1-1-1-1-1-1-1-1-1-10-1-16-1-25-1/retained_before_repair_2.
Compiler register cap 96 retained from the source IR. One compile per variant,
two warmups and five measured executions; the existing verifier ranks by the
trimmed mean. Source files and benchmark IR were frozen in the output folder.

| Single change from baseline | GFLOPS | Change |
|---|---:|---:|
| Baseline | 6489.85 | 0.00% |
| No padding | 6480.49 | -0.14% |
| Only A padding | 5682.40 | -12.44% |
| Hoist A fragment load outside wn loop | 6126.42 | -5.60% |
| Scalar store | 6453.70 | -0.56% |
| Guarded float4 store rewrite | 6490.49 | +0.01% |
| Additional ping-pong reuse barrier | 6268.56 | -3.41% |
| Baseline repeat | 6457.08 | -0.51% |

All eight cases pass compile, correctness and the existing runtime safety
oracle. This is not an exhaustive proof for other shapes or all races.
A/B padding already matched baseline. General pipeline reordering was not
performed because the tool cannot prove its synchronization requirements.

The store change is below observed baseline drift; no meaningful improvement
was demonstrated. None of these variants was promoted or copied to skeleton.
This comparison does not establish that new selection priors improve the
end-to-end generator; that requires a fresh generation run. The full app and
LLM generation were not run as part of this repair.

Artifacts: gemm_code/controlled_comparison/current_512_ir_repair_20260922/
comparison.json, frozen_sources.json and frozen_benchmark_ir.json.

## Reproduction

From the directory containing SCOPE:

```text
python -m SCOPE.compare_kernel_controls --source <accepted-code-directory> --ir <accepted-code-directory>/verified_ir.json --output <new-separate-directory> --matrix-size 512 512 512 --build-platform windows --benchmark-runs 5 --warmup-runs 2
```

Use --build-platform linux on Linux. Add --prepare-only to save source variants
without compiling or running them. Existing nonempty output directories are
rejected to avoid mixing measurements. The input directory is never edited.
