# Shape specialization

## Tune a specific existing CUDA implementation

`tune.py` operates only on the selected existing source and makes no LLM
calls. It does not require an archived family or existing verification JSON.
Run from the parent directory of SCOPE:

```sh
python -m SCOPE.tune --backend cuda --source /root/scope/SCOPE/gemm_code/code/chain.example --shapes 1024 2048 --build-platform linux
python -m SCOPE.tune --source /root/scope/SCOPE/gemm_code/code/chain.example/cuda_kernel.cuh --matrix-size 1024 2048 512 --build-platform linux --max-candidates 60
python -m SCOPE.tune --backend cpu --source /root/scope/SCOPE/gemm_code/cpu_code/chain.example --matrix-size 512 512 512 --build-platform linux --benchmark-runs 10
```

Source directory needs the benchmark main.cpp and its headers. By default, test
1024 cubed, all legal strategy-library Block/Warp/Thread/iteration tuples over 3 rounds, compile each candidate
once, then benchmark 5 runs with 2 warmups. `--output PATH` selects a fresh output
directory; otherwise use `gemm_code/shape_tuning/<run>`. Each shape has search.json,
top_3_results.json, candidate directories and source copies in top_3/rank_1..3.
`--aggregation mean` ranks candidates by the arithmetic mean of every measured
run; `--aggregation trimmed_mean` drops one minimum and one maximum first.
Failures are recorded and subsequent candidates continue. Existing source files
are unchanged. `--target-ratio 0.8` optionally stops after reaching 80% of cuBLAS.

The default `--max-candidates 0` imposes no candidate-count limit. Enumeration
is hierarchical: Block, divisible Warp, Thread, then every legal WMITER/WNITER.
It requires `(WMITER/TM)*(WNITER/TN) == warp_size` and exact divisibility of
WM/WN by WMITER/WNITER. No occupancy estimate or diversity shortlist removes
otherwise legal tuples. Compiler/runtime checks still reject unsupported source
configurations and resource usage. This is exhaustive only within the sizes
declared in the strategy library, not every positive integer tile size.
`candidate_manifest.json` records the entire legal domain. A positive
`--max-candidates` is an explicit partial-test budget, reported separately from
pool exhaustion. Do not set `--target-ratio` when all candidates must be tested.

This source-preserving mode requires unique numeric launch declarations for BM,
BN, BK, WM, WN, TM, TN, WMITER and WNITER between LAUNCH_CONFIG markers. It preserves
the original compute/load/pipeline code, so only parameter-dependent layout and
mapping change automatically. Arbitrary structural layout, pipeline and split-K
transformations are not inferred. Unsupported parameter contracts are reported
after testing the baseline; an unrelated generated kernel is never substituted.
Optional `--ir PATH` supplies source metadata; hardware is reprobed on this machine.

## Automatic reuse in app.py

After terminal verification and unlock, `app.py` now tunes up to three accepted
GPU seeds with LLM-proposed coupled Tile ranges and merges the passing
results into final selection before publishing the skeleton. Original candidates
remain available on failures. The shape-reuse early-return route also runs this
step. CPU is unchanged. Outputs use fresh `gemm_code/shape_tuning/terminal` run
directories. Defaults (an omitted section uses these values):

```yaml
terminal_tile_tuning:
  enabled: true
  max_seeds: 3
  max_candidates: 0  # all legal tuples WITHIN the LLM ranges
  max_combinations: 256  # reject oversized plans; never silently truncate
  timeout_seconds: 180
```

Benchmark repetition settings come from the normal terminal configuration. Each
seed uses an LLM request containing current kernel code (not main.cpp), selected
strategies, IR/resource information and current performance. Invalid JSON or
range validation failures get exactly one targeted correction request. Limits
(16 groups, 32 tuples per array, 256 expanded combinations by default) are
included in the prompt. Raw JSON-mode responses and validation outcomes are saved
under seed_N.planning, including responses that cannot be parsed. Request/network
failures remain failures rather than being treated as weak performance.
It returns groups
of Block/Warp/Thread ranges, not code. Each group is independently expanded to
all mapping-valid iteration layouts and deduplicated. Planning errors preserve
original candidates; they never trigger whole-library enumeration. Empty plans
only retest the baseline. Positive max_candidates explicitly limits coverage.
Inspect seed_N.tile_plan.json, candidate_manifest.json and search.json. The
standalone tune CLI still supports whole-library enumeration explicitly.

`app.py` defaults to `--shape-reuse auto` for CUDA. With no compatible archive,
the existing LLM generation flow runs and archives its top three passing results.
Archives live in `data/kernel_families`, outside normal startup cleanup.

```sh
python -m SCOPE.app --build-platform linux --matrix-size 512 512 512
python -m SCOPE.app --build-platform linux --matrix-size 1024 1024 1024
python -m SCOPE.app --build-platform linux --matrix-size 2048 2048 2048
```

Argument order is **M N K**. Existing DEFAULT_DESCRIPTION still works when the
argument is omitted. CPU follows the existing generation flow.

Configuration in conf.yaml:

```yaml
shape_reuse:
  enabled: true
  target_ratio: 0.8
  max_rounds: 3
  candidates_per_round: 12
  max_families: 3
  split_k_slices: [2, 4, 8]
  max_workspace_bytes: 268435456
  timeout_seconds: 180
```

Use `--shape-reuse off` for the original generation flow, or `--shape-reuse only`
to measure deterministic candidates without making LLM calls. In only mode,
an unmet target is reported explicitly and the best correct result is retained.
Import an existing result with `--reuse-family PATH`; PATH must contain source
files and `verified_ir.json` with passing compile, correctness, runtime results.
Import runs before cleanup. Imported evidence does not replace revalidation.

Round 0 retests archived implementations on the new shape. Round 1 searches
Block/Warp/Thread tuples, derives lane ownership, and instantiates either a
parameterized seed (unique LAUNCH_CONFIG constants required) or the explicit
built-in SIMT family. Round 2 changes shared A orientation, padding and unroll,
updating producers and consumers together; it also tests static double buffering
and cp.async when the hardware reports support. Round 3 tries split-K using grid.z,
separate partial outputs and one reduction kernel. Alpha/beta is applied once.
Workspace allocation/free is per call and is included in measured execution;
this first executor may lose to direct GEMM because of those costs.

Every candidate uses a fresh source instance, compiles once and uses the existing
benchmark repetition settings. A failure cannot overwrite the archived source.
No LLM or Patch JSON is needed for deterministic instances. Lightweight records
and verifier IR are kept under `gemm_code/shape_code/<shape>.<run>/`.
`results/check/shape_reuse_result.json` points to the current search results.
Final Top-3 includes both reuse and new LLM-generated passing candidates.

If three bounded rounds miss target_ratio relative to the same run's cuBLAS,
the existing LLM workflow receives current-shape measurement feedback and runs
again. Missing cuBLAS metrics cannot count as reaching the target.

Limits: the built-in family implements FP32 row-major NN, scalar staged loads,
layout variants, static double buffering, 4-byte cp.async and split-K workspace
reduction. It does not implement float4, persistent CTA or Stream-K. Archived code can retain those optimizations
when retested unchanged or when its launch constants can be specialized.
Generated arbitrary layouts cannot safely be rewritten by constant substitution:
these changes remain separate built-in-family candidates or trigger the LLM
fallback. Hardware identity and semantic compatibility are required for reuse.
