"""Aggregate SCOPE RQ2/RQ3 run records into paper-ready CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_records(roots: list[Path], filename: str) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    seen = set()
    for root in roots:
        for path in sorted(root.resolve().rglob(filename)):
            if path in seen:
                continue
            seen.add(path)
            records.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return records


def rq2_rows(records: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for path, record in records:
        outcome = record.get("outcome", {}) or {}
        experiment = record.get("experiment", {}) or {}
        counters = record.get("shadow_checks", {}) or {}
        rows.append({
            "run_id": record.get("run_id"),
            "variant": record.get("variant"),
            "backend": experiment.get("backend"),
            "shape": shape_label(experiment.get("problem")),
            "llm_provider": experiment.get("llm_provider"),
            "llm_model": experiment.get("llm_model"),
            "terminal_chains": outcome.get("terminal_chain_count"),
            "verified_chains": outcome.get("verified_terminal_chain_count"),
            "accepted_chains": outcome.get("accepted_terminal_chain_count"),
            "acceptance_rate": outcome.get("acceptance_rate"),
            "best_gflops": outcome.get("best_gflops"),
            "elapsed_seconds": outcome.get("elapsed_seconds"),
            "llm_requests": (record.get("llm_usage", {}) or {}).get("request_count"),
            "llm_tokens": (record.get("llm_usage", {}) or {}).get("total_tokens"),
            "shadow_pre_rejected": counters.get("local_precondition.rejected", 0),
            "shadow_post_rejected": counters.get("local_postcondition.rejected", 0)
                + counters.get("local_patch_postcondition.rejected", 0),
            "shadow_stage_rejected": counters.get("stage_constraint.rejected", 0)
                + counters.get("stage_requirement.rejected", 0),
            "record_path": str(path),
        })
    return rows


def rq3_rows(records: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for path, record in records:
        phase2 = record.get("phase2_feedback_optimization", {}) or {}
        if phase2.get("triggered"):
            rows.append({
                "phase": "cuda_phase2_feedback",
                "backend": "cuda",
                "shape": "pipeline_target",
                "seed_gflops": phase2.get("phase1_best_gflops"),
                "winner_gflops": phase2.get("phase2_best_gflops"),
                "gain_gflops": phase2.get("gain_gflops"),
                "gain_percent": phase2.get("gain_percent"),
                "elapsed_seconds": phase2.get("elapsed_seconds"),
                "tested_candidates": phase2.get("terminal_candidate_count"),
                "accepted_candidates": phase2.get("accepted_candidate_count"),
                "compile_attempts": (phase2.get("search_cost", {}) or {}).get("compilation_count"),
                "benchmark_process_runs": None,
                "stop_reason": None,
                "record_path": str(path),
            })
        phase3 = record.get("phase3_constrained_reinstantiation", {}) or {}
        for item in phase3.get("shapes", []) or []:
            cost = item.get("search_cost", {}) or {}
            rows.append({
                "phase": "phase3_constrained_reinstantiation",
                "backend": phase3.get("backend"),
                "shape": shape_label(item.get("shape")),
                "seed_gflops": item.get("seed_gflops"),
                "winner_gflops": item.get("winner_gflops"),
                "gain_gflops": item.get("gain_gflops"),
                "gain_percent": item.get("gain_percent"),
                "elapsed_seconds": cost.get("elapsed_seconds", item.get("wallclock_seconds")),
                "tested_candidates": cost.get("tested_candidates"),
                "accepted_candidates": cost.get("accepted_candidates"),
                "compile_attempts": cost.get("compile_attempts"),
                "benchmark_process_runs": cost.get("benchmark_process_runs"),
                "stop_reason": item.get("stop_reason"),
                "record_path": str(path),
            })
    return rows


def standalone_tune_row(path: Path, record: dict[str, Any]) -> dict[str, Any] | None:
    results = record.get("results", []) or []
    accepted = [item for item in results if item.get("accepted") and item.get("gflops") is not None]
    seed = next((item for item in results if item.get("candidate_id") == 0), None)
    if not seed or seed.get("gflops") is None or not accepted:
        return None
    winner = max(accepted, key=lambda item: float(item["gflops"]))
    seed_gflops = float(seed["gflops"])
    winner_gflops = float(winner["gflops"])
    gain = winner_gflops - seed_gflops
    benchmark_runs = sum(
        int((item.get("performance", {}) or {}).get("benchmark_successful_runs", 0) or 0)
        for item in results
    )
    backend = record.get("backend")
    if not backend and "cuda" in str(record.get("mode", "")).lower():
        backend = "cuda"
    return {
        "phase": "phase3_constrained_reinstantiation",
        "backend": backend,
        "shape": shape_label(record.get("shape")),
        "seed_gflops": seed_gflops,
        "winner_gflops": winner_gflops,
        "gain_gflops": gain,
        "gain_percent": (100.0 * gain / seed_gflops) if seed_gflops else None,
        "elapsed_seconds": record.get("elapsed_seconds"),
        "tested_candidates": record.get("tested_candidate_count"),
        "accepted_candidates": record.get("accepted_candidate_count"),
        "compile_attempts": record.get("compile_attempt_count"),
        "benchmark_process_runs": benchmark_runs,
        "stop_reason": record.get("stop_reason"),
        "record_path": str(path),
    }


def shape_label(shape: Any) -> str | None:
    if not isinstance(shape, dict):
        return None
    values = [shape.get(key) for key in ("M", "N", "K")]
    return "x".join(map(str, values)) if all(value is not None for value in values) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Run/result roots to scan recursively.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument(
        "--tune-search",
        action="append",
        type=Path,
        default=[],
        help="Standalone tune.py search.json to include as a Phase 3 row; repeat as needed.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rq2 = rq2_rows(load_records(args.roots, "rq2_ablation_record.json"))
    rq3 = rq3_rows(load_records(args.roots, "rq3_optimization_record.json"))
    for path in args.tune_search:
        resolved = path.resolve()
        row = standalone_tune_row(resolved, json.loads(resolved.read_text(encoding="utf-8")))
        if row:
            rq3.append(row)
    write_csv(args.output_dir / "rq2_ablation_runs.csv", rq2)
    write_csv(args.output_dir / "rq3_optimization_runs.csv", rq3)
    summary = {"rq2_run_count": len(rq2), "rq3_row_count": len(rq3)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
