# RQ2/RQ3 ablation findings (RTX 4060 Ti, FP32 GEMM 512^3)

## Experiment coverage

Eight configurations are represented. The historical `Full` row is retained for RQ2
provenance. Four construction ablations are fresh runs. `w/o Feedback Optimization`
is freshly evaluated from the archived Full Phase-1 checkpoint. The formal parameter-
search ablation is a paired exhaustive run from one shared Phase-2 checkpoint: its
unchanged seed is `w/o Parameter Search`, and its best accepted re-instantiation is
`Full Parameter Search (r4)`.

| Configuration | Phase-1 accepted | Phase-1 GFLOPS | Final GFLOPS | LLM requests | Tokens | Measured incremental time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Full | 3/3 | 6768.29 | 10577.16 | archived | archived | archived |
| w/o State-in-Prompt | 1/3 | 5642.36 | 5985.93 | 162 | 2.068M | 2542.10 |
| Intent-only Control (interface failure) | 0/0 | - | - | 4 | 0.025M | 40.78 |
| w/o Local Checks | 3/3 | 5902.79 | 7469.91 | 172 | 3.542M | 2900.96 |
| w/o Stage Ordering | 2/3 | 6045.23 | 6911.72 | 184 | 3.865M | 4762.93 |
| w/o Feedback Optimization | 3/3 | 6768.29 | 6478.41 | 0 | 0 | 113.36 |
| w/o Parameter Search | 3/3 | 6768.29 | 8897.07 | 0 | 0 | 7.17 |
| Full Parameter Search (r4) | 3/3 | 6768.29 | 9853.71 | 0 | 0 | 11974.96 |

## RQ2 answer

- Structured observed state is important for robust construction. Removing it reduced
  terminal acceptance from 100% to 33.3%. The original intent-only 0/0 result is an
  interface-failure record, not primary evidence: symbolic mapping placeholders
  reached typed IR fields before the construction behavior could be evaluated.
- Local checks mainly reduce delayed error propagation and search cost in this run.
  Removing them still produced three accepted terminal kernels, but consumed 3.542M
  tokens and finished 29.4% below Full after optimization.
- Fixed stage ordering improved reliability and efficiency. Dynamic LLM ordering moved
  Pipeline before Mapping/Reordering, accepted only two of three terminal chains, took
  4762.9 seconds, and finished 34.7% below Full.
- These results support the claim that acceptance comes from the combination of explicit
  state, local checks, and staged constraints rather than from unconstrained LLM repair.

### Interface-repaired Intent-only replication

The Intent-only adapter materializes only mapping fields uniquely determined by the
selected BlockTile, WarpTile, and hardware warp size. It does not expose source-derived
observations, reuse Full decisions, or change the search and acceptance budgets.

| Run | Terminal accepted | Phase-1 GFLOPS | Final GFLOPS | Requests | Tokens | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 2/3 | 6241.96 | 7755.74 | 151 | 1.995M | 2462.91 |
| r2 | 0/2 | - | - | 179 | 2.336M | 2843.41 |
| r3 | 0/3 | - | - | 178 | 2.414M | 2960.07 |

All three repaired runs reached terminal verification, eliminating the original 0/0
interface failure. Across the replications, 2 of 8 terminal candidates were accepted,
and only 1 of 3 runs produced an accepted output. Every terminal candidate compiled;
the unsuccessful runs instead failed correctness or runtime safety, including candidates
that did not reach correctness after a runtime-safety failure. The successful run reached
6241.96 GFLOPS after construction and 7755.74 GFLOPS after feedback optimization
(+24.25%). This is evidence that intent alone can occasionally construct a valid kernel,
but is substantially less reliable when source observations are withheld.

## RQ3 answer

- For `w/o State-in-Prompt`, CUDA Phase 2 improved the Phase-1 winner by 6.09%.
- For `w/o Local Checks`, CUDA Phase 2 improved it by 26.55%.
- For `w/o Stage Ordering`, CUDA Phase 2 improved it by 14.33%.
- The earlier 32-candidate prefix was not a complete parameter-search ablation. It was
  biased toward the low-end `BM=32, BN=32, BK=8` neighborhood and is retained only as
  a budget-limited diagnostic.
- The formal r4 run exhausted all 1710 candidate records with no budget omission.
  Relative to the unchanged seed, constrained parameter re-instantiation improved
  performance from 8897.07 to 9853.71 GFLOPS, a **10.752%** gain.
- Exhaustive search cost 11974.96 seconds (3.326 hours). Of 1710 records, 1560 compiled,
  792 were runtime-safe, and 79 passed the complete acceptance checks (78 transformed
  candidates plus the unchanged seed).
- Both Phase-3 parameter configurations made zero LLM requests and consumed zero LLM
  tokens. `w/o Parameter Search` still required 7.17 seconds to compile, validate, and
  benchmark the unchanged seed; `w/o Feedback Optimization` required 113.36 seconds
  because its 32-candidate constrained search remained enabled.
- The best configuration was `BM=64, BN=128, BK=16, WM=32, WN=32, TM=8, TN=4,
  WMITER=32, WNITER=32`. These results show measurable Phase-3 benefit but also a high
  validation and search cost, motivating better ordering or pruning without losing this
  neighborhood.

## Validity note

The archived Full result predates the ablation instrumentation changes, while the four
construction ablations are fresh runs. The RQ2 table therefore remains a reuse-based
comparison. The parameter-search claim is stronger because its seed and winner come
from the same r4 execution and benchmark protocol, but it is still a single run on one
shape and should be replicated before a camera-ready causal claim.

The repaired Intent-only result is based on three independent runs. Its historical 0/0
record is retained only to document the interface defect and is excluded from the
repaired-run acceptance estimate. The 1/3 run success rate and 2/8 pooled terminal
acceptance rate should be reported together because they capture different reliability
levels; neither should be presented as a deterministic effect estimate.

The exact active input, reference, tolerance, math-mode, threading, and timing
settings are recorded in `EXPERIMENT_PROTOCOL_AUDIT.md`. In particular, the CUDA
generation verifier and standalone `benchmark.py` have different max-absolute-error
policies and must not be described as one protocol.
