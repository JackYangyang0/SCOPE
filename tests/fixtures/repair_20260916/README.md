# Fixed CUDA repair regressions

Frozen from the 2026-09-16 23:47 terminal run, before repair. These are failed
inputs, not reference implementations or performance targets. Keep them immutable.

- case1: chain.2-1-1-1-1-1-1-1-1-1-1-1-1. Invalid cooperative load counts/strides.
- case2: chain.3-1-1-1-1-1-1-1-1-1-1-1. A vector mapping, incomplete B coverage,
  and unaligned float4 shared stores with padded rows.
- case3: chain.3-1-1-1-1-1-1-1-1-1-2-1. Only half of the B tile loaded.

Run from the repository parent:

```text
python -m SCOPE.tests.run_fixed_repair_regression
```

Each run creates a fresh results/code/fixed_repair_TIMESTAMP directory, retains
the original inputs, compiles the repaired copies, and uses the existing complete
terminal verification path including locked strategies. Reports distinguish
dynamic correctness from final acceptance. No historical ranking is overwritten.

The narrow deterministic repairs support recognized row-major, non-transposed,
full-tile float4 schedules only. Unsupported code goes through the LLM repair
path; matching these three fixtures does not establish general repair success.

The independent fractional checker runs the same 1024 shape with signed fractional
inputs, alpha=0.75 and beta in {0, 1, -0.5} against pedantic cuBLAS. It is a
Windows/sm_89 local regression utility, not a portable hardware benchmark.
