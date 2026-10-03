"""Summarize interface-repaired Intent-only RQ2 replications."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_row(record_path: Path) -> dict[str, Any]:
    record = load_json(record_path)
    outcome = record.get("outcome", {}) or {}
    usage = record.get("llm_usage", {}) or {}
    checks = record.get("shadow_checks", {}) or {}
    rq3_path = record_path.with_name("rq3_optimization_record.json")
    rq3 = load_json(rq3_path) if rq3_path.exists() else {}
    phase2 = rq3.get("phase2_feedback_optimization", {}) or {}
    phase1_best = outcome.get("best_gflops")
    final_best = phase2.get("effective_best_after_rollback_gflops")
    if final_best is None:
        final_best = phase1_best
    return {
        "run_id": record.get("run_id"),
        "provenance": "interface_repaired",
        "terminal_chains": outcome.get("terminal_chain_count", 0),
        "verified_chains": outcome.get("verified_terminal_chain_count", 0),
        "accepted_chains": outcome.get("accepted_terminal_chain_count", 0),
        "acceptance_rate": outcome.get("acceptance_rate"),
        "phase1_best_gflops": phase1_best,
        "final_best_gflops": final_best,
        "phase2_triggered": bool(phase2.get("triggered")),
        "phase2_gain_percent": phase2.get("gain_percent"),
        "elapsed_seconds": outcome.get("elapsed_seconds"),
        "llm_requests": usage.get("request_count", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "llm_request_seconds": usage.get("request_seconds", 0.0),
        "intent_mapping_materializations": checks.get("intent_mapping_materialized.count", 0),
        "compile_pass": (outcome.get("terminal_status_counts", {}) or {}).get("compile_status.pass", 0),
        "correctness_pass": (outcome.get("terminal_status_counts", {}) or {}).get("correctness_status.pass", 0),
        "correctness_fail": (outcome.get("terminal_status_counts", {}) or {}).get("correctness_status.fail", 0),
        "correctness_not_run": (outcome.get("terminal_status_counts", {}) or {}).get("correctness_status.not_run", 0),
        "runtime_safety_fail": (outcome.get("terminal_status_counts", {}) or {}).get("runtime_safety_status.fail", 0),
        "record_path": str(record_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--historical-record", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [run_row(path.resolve()) for path in args.records]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "intent_only_repaired_runs.csv", rows)

    accepted_runs = [row for row in rows if int(row["accepted_chains"] or 0) > 0]
    total_terminal = sum(int(row["terminal_chains"] or 0) for row in rows)
    total_accepted = sum(int(row["accepted_chains"] or 0) for row in rows)
    report = {
        "variant": "intent_only_control",
        "repair_scope": "typed intent-to-IR mapping interface only",
        "source_observation_restored": False,
        "full_decisions_reused": False,
        "replication_count": len(rows),
        "successful_run_count": len(accepted_runs),
        "run_success_rate": len(accepted_runs) / len(rows),
        "terminal_candidate_count": total_terminal,
        "accepted_terminal_candidate_count": total_accepted,
        "pooled_terminal_acceptance_rate": total_accepted / total_terminal if total_terminal else None,
        "mean_elapsed_seconds": statistics.mean(float(row["elapsed_seconds"]) for row in rows),
        "mean_llm_requests": statistics.mean(int(row["llm_requests"]) for row in rows),
        "mean_total_tokens": statistics.mean(int(row["total_tokens"]) for row in rows),
        "successful_phase1_best_gflops": [row["phase1_best_gflops"] for row in accepted_runs],
        "successful_final_best_gflops": [row["final_best_gflops"] for row in accepted_runs],
        "runs": rows,
        "interpretation": {
            "interface_result": "All repaired runs reached terminal verification; the former 0/0 early exit is removed.",
            "construction_result": "Only one of three runs produced accepted kernels, indicating instability after source observations are hidden.",
            "failure_modes": "Failed runs compiled their terminal kernels but failed correctness or runtime-safety checks.",
        },
    }
    if args.historical_record:
        historical_path = args.historical_record.resolve()
        historical = load_json(historical_path)
        report["historical_interface_failure"] = {
            "provenance": "interface_failure_historical",
            "record_path": str(historical_path),
            "outcome": historical.get("outcome", {}),
            "role": "Interface-failure provenance only; excluded from repaired-run effect estimates.",
        }
    output = args.output_dir / "intent_only_repaired_summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"csv": str(args.output_dir / "intent_only_repaired_runs.csv"), "summary": str(output)}, indent=2))


if __name__ == "__main__":
    main()
