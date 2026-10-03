# Ampere FP32 SIMT strategy support

The implementation is capability-based, not an A800 name-based preset. CPU code,
initial skeleton files and existing experiment outputs are not migrated.

## Implemented

- Capability-class execution profiles distinguish CC 8.0 data-center Ampere,
  other Ampere devices, Ada, Hopper and Blackwell-family targets without matching
  product names. Runtime-probed limits override architecture defaults.
- Joint tiling candidates estimate resident CTAs limited by threads, blocks,
  registers and shared memory. They also record active warps, occupancy, SM
  coverage, CTA waves, last-wave utilization and arithmetic intensity.
- Tiling prompts receive a bounded, architecture-ranked legal tuple list. Stage
  pruning uses the architecture score when no target measurement is available;
  actual compile/correctness/performance results remain authoritative later.
- CUDA logical-device attribute probing: ordinary/opt-in shared limits, shared and
  registers per SM, L2 persistence limits and compute capability. Failed probes
  remain unavailable; limits are not inferred from product names.
- Physical per-stage A/B shapes, padding and dtype determine pipeline allocation.
  Replacing two stages with three uses three stages, not six. Auxiliary allocation
  is reported separately. Patch IR and stage resource derivation share this model.
- Two/three/four-stage cp.async candidates have explicit IR predicates, protocol
  obligations, dynamic correctness/runtime checks and instruction/API evidence.
  Instruction presence is NOT a proof of correct synchronization or full coverage.
- Available compute-sanitizer racecheck/synccheck run after a correct async kernel.
  Unavailable tools and timeouts are recorded as not_run/inconclusive, not pass.
- Dynamic shared-memory local patches preserve the current layout. The program
  configures the actual specialization's opt-in and third launch argument. The
  supported standard wrapper has exactly one named-grid/block GEMM launch.
- Carveout 0/50/100 preferences are host-configuration alternatives. They do not
  imply exact hardware partition sizes or guaranteed speedups.
- A pipeline exceeding the ordinary limit can be planned with a required dynamic
  allocation strategy when measured device opt-in capacity permits it.
- Graph fallbacks descend from four to three to two stages, then synchronous
  buffering. Batch construction loads complete library records and retains checks.
- PTXAS register-budget sweep tests current/default, 64, 96 and 128 limits, deduped.
  Each variant is compiled once, then uses the existing repeated-run benchmark.
  Only correct, runtime-safe variants with finite positive latency are rankable.
  The best executable is copied to the ordinary build destination. The reports
  `build/resource_sweep.json` and `build/selected_build_config.json` retain flags.
  Reuse the selected flags when manually rebuilding the winning source.

## Intentionally unavailable

- Split-K/Stream-K: require workspace management, multiple launches, exact K
  partitioning, one alpha/beta epilogue, and complete-operation timing.
- L2 access-policy windows: require ownership/restoration of stream and context
  settings, queried limits, and handling of MIG/MPS restrictions.
- Launch-bounds tuning: needs a kernel-declaration patch boundary and a real
  launch-bounds executor. The enabled PTXAS sweep currently tunes register budgets,
  NOT launch bounds.
- Bank-conflict analysis: needs a dedicated address/profiler analysis pass.

These entries remain excluded_default, with activation requirements in the
library. They must not be enabled by changing a phase label alone. WarpAwareSwizzle
aliases canonical XOR-layout strategies to avoid duplicate candidates.

## Validation

Regression tests exercise accounting, catalog/index/graph consistency, batch
contracts, async evidence, host launch materialization and mocked compiler sweeps.
They do not establish performance or CUDA correctness on A800 hardware. Run the
normal app workflow on the target GPU; retain strict FP32 comparison settings.

Reapply the targeted catalog migration with:

```text
python -m SCOPE.utils.upgrade_ampere_strategies
```

Run this from the directory containing SCOPE. This migration does not reset other
strategies or the initial IR template.
