# RQ2/RQ3 Ablation and Cost Records

## RQ2: Why does SCOPE generate accepted code?

Run four variants with the same model, hardware, shape, seed/configuration, and benchmark settings:

1. `full`: structured generation state + local checks + stage constraints.
2. `minus_structured_state`: hide tiling/mapping/memory/pipeline/resource/history state from LLM prompts while retaining it internally for execution and measurement.
3. `minus_local_checks`: compute pre/post/local checks in shadow mode, but do not let them reject candidates.
4. `minus_stage_constraints`: compute stage required-field/completion predicates in shadow mode, but do not let them reject paths.

Example commands:

```bash
python -m SCOPE.app --build-platform linux --matrix-size 1024 1024 1024 --ablation full --ablation-run-id gpu5090-1024-r1
python -m SCOPE.app --build-platform linux --matrix-size 1024 1024 1024 --ablation minus_structured_state --ablation-run-id gpu5090-1024-r1
python -m SCOPE.app --build-platform linux --matrix-size 1024 1024 1024 --ablation minus_local_checks --ablation-run-id gpu5090-1024-r1
python -m SCOPE.app --build-platform linux --matrix-size 1024 1024 1024 --ablation minus_stage_constraints --ablation-run-id gpu5090-1024-r1
```

Use at least three independent repetitions per hardware/shape/variant. Report:

- terminal chains, compiled/verified chains, accepted chains, and acceptance rate;
- compile, correctness, and runtime-safety failures;
- shadow pre/post/stage rejections (what the removed component would have rejected);
- best accepted GFLOPS as a secondary guard against accepting trivial but slow code;
- wall-clock time, LLM request count, and prompt/completion/total tokens.

Each run writes the latest record to `results/check/rq2_ablation_record.json`. When `--ablation-run-id` is supplied, it also archives both RQ records under `results/experiments/runs/<run-id>/<variant>/`. Shadow checks are capped at 2,000 detailed events, while aggregate counters are complete.

## RQ3: What gain and cost come from feedback and reinstantiation?

CUDA Phase 2 is measured as the difference between the best accepted Phase 1 terminal kernel and the best accepted feedback-unlock kernel. Report absolute GFLOPS, percentage gain, feedback rounds, tested/accepted candidates, compilation attempts, wall-clock time, and LLM usage.

CPU/CUDA Phase 3 is measured independently for every target shape. Candidate `0` is the unchanged verified seed instantiated at that target shape; the winner is compared with this per-shape seed, not with the source shape's old measurement. Report tested/accepted/failed candidates, legal and budget-omitted candidates, compilation attempts, benchmark process runs, rounds, stop reason, and wall-clock time.

Each run writes `results/check/rq3_optimization_record.json`. Aggregate copied result directories with:

```bash
python scripts/summarize_rq_experiments.py results/experiments/runs --output-dir results/experiments/tables
```

This creates `rq2_ablation_runs.csv`, `rq3_optimization_runs.csv`, and `summary.json`. Do not aggregate old result trees produced before these schemas were introduced, because they lack per-shape seed and cost records.
