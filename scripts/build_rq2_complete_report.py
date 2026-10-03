from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-id", default="rq2-4060ti-512-r1")
    args = parser.parse_args()

    repo = args.repo.resolve()
    run_root = repo / "results" / "experiments" / "runs" / args.run_id
    output_dir = repo / "results" / "experiments" / f"{args.run_id}-summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    historical = repo / "4060ti" / "qwen3.5-397b-a17b" / "512"
    terminal_records = [load_json(path) for path in historical.glob("code/**/terminal_chain.json")]
    accepted_terminal = [
        item for item in terminal_records
        if nested(item, "verification", "compile_status") == "pass"
        and nested(item, "verification", "correctness_status") == "pass"
        and nested(item, "verification", "runtime_safety_status") == "pass"
    ]
    phase1_best = max(
        (float(nested(item, "verification", "gflops_mean", default=0.0)) for item in accepted_terminal),
        default=0.0,
    )
    unlock_path = historical / "results" / "check" / "performance_unlock_summary.json"
    unlock = load_json(unlock_path)
    phase2_best = float(unlock.get("best_gflops") or 0.0)

    fresh_variants = [
        "minus_state_in_prompt",
        "intent_only_control",
        "minus_local_checks",
        "minus_stage_ordering",
    ]
    repaired_intent_path = output_dir / "intent_only_repaired_summary.json"
    repaired_intent = load_json(repaired_intent_path) if repaired_intent_path.exists() else None
    rows: list[dict[str, Any]] = []
    for variant in fresh_variants:
        if variant == "intent_only_control" and repaired_intent:
            successful_phase1 = repaired_intent.get("successful_phase1_best_gflops", []) or []
            successful_final = repaired_intent.get("successful_final_best_gflops", []) or []
            rows.append(
                {
                    "variant": variant,
                    "provenance": "interface_repaired_3_run_aggregate",
                    "phase1_terminal_chains": repaired_intent.get("terminal_candidate_count", 0),
                    "phase1_accepted_chains": repaired_intent.get("accepted_terminal_candidate_count", 0),
                    "phase1_acceptance_rate": repaired_intent.get("pooled_terminal_acceptance_rate"),
                    "phase1_best_gflops": max(successful_phase1, default=None),
                    "final_gflops": max(successful_final, default=None),
                    "elapsed_seconds": repaired_intent.get("mean_elapsed_seconds"),
                    "llm_requests": repaired_intent.get("mean_llm_requests"),
                    "llm_tokens": repaired_intent.get("mean_total_tokens"),
                    "source": str(repaired_intent_path),
                }
            )
            continue
        record_path = run_root / variant / "rq2_ablation_record.json"
        record = load_json(record_path)
        pipeline = record.get("outcome") or {}
        rq3_path = run_root / variant / "rq3_optimization_record.json"
        rq3 = load_json(rq3_path) if rq3_path.exists() else {}
        final_values = [pipeline.get("best_gflops")]
        final_values.append(nested(rq3, "phase2_feedback_optimization", "effective_best_after_rollback_gflops"))
        final_values.extend(
            item.get("winner_gflops")
            for item in nested(rq3, "phase3_constrained_reinstantiation", "shapes", default=[])
            if isinstance(item, dict)
            and all(int((item.get("shape") or {}).get(axis, 0)) == 512 for axis in ("M", "N", "K"))
        )
        final_gflops = max((float(value) for value in final_values if value is not None), default=None)
        rows.append(
            {
                "variant": variant,
                "provenance": "fresh_run",
                "phase1_terminal_chains": pipeline.get("terminal_chain_count", 0),
                "phase1_accepted_chains": pipeline.get("accepted_terminal_chain_count", 0),
                "phase1_acceptance_rate": pipeline.get("acceptance_rate"),
                "phase1_best_gflops": pipeline.get("best_gflops"),
                "final_gflops": final_gflops,
                "elapsed_seconds": pipeline.get("elapsed_seconds"),
                "llm_requests": nested(record, "llm_usage", "request_count", default=0),
                "llm_tokens": nested(record, "llm_usage", "total_tokens", default=0),
                "source": str(record_path),
            }
        )

    full_base = {
        "phase1_terminal_chains": len(terminal_records),
        "phase1_accepted_chains": len(accepted_terminal),
        "phase1_acceptance_rate": (len(accepted_terminal) / len(terminal_records)) if terminal_records else None,
        "phase1_best_gflops": phase1_best,
        "elapsed_seconds": None,
        "llm_requests": None,
        "llm_tokens": None,
    }
    rows.append(
        {
            "variant": "full",
            "provenance": "historical_reuse",
            **full_base,
            "final_gflops": phase2_best,
            "source": str(unlock_path),
        }
    )
    parameter_search_path = (
        repo / "results" / "experiments" / "runs" / "rq2-4060ti-512-r4"
        / "parameter_search" / "512x512x512" / "search.json"
    )
    parameter_search = load_json(parameter_search_path)
    parameter_results = parameter_search.get("results", []) or []
    seed = next(item for item in parameter_results if item.get("candidate_id") == 0)
    accepted_parameter_results = [
        item for item in parameter_results
        if item.get("accepted") and item.get("gflops") is not None
    ]
    parameter_winner = max(accepted_parameter_results, key=lambda item: float(item["gflops"]))
    rows.append(
        {
            "variant": "minus_parameter_search",
            "provenance": "paired_seed_from_full_parameter_search_r4",
            **full_base,
            "elapsed_seconds": seed.get("evaluation_seconds"),
            "llm_requests": 0,
            "llm_tokens": 0,
            "final_gflops": seed.get("gflops"),
            "source": str(parameter_search_path),
        }
    )
    rows.append(
        {
            "variant": "full_parameter_search_r4",
            "provenance": "exhaustive_from_shared_full_phase2_checkpoint",
            **full_base,
            "elapsed_seconds": parameter_search.get("elapsed_seconds"),
            "llm_requests": 0,
            "llm_tokens": 0,
            "final_gflops": parameter_winner.get("gflops"),
            "source": str(parameter_search_path),
        }
    )

    phase3_path = run_root / "minus_feedback_optimization" / "phase3_from_full_p1" / "summary.json"
    phase3_raw = json.loads(phase3_path.read_text(encoding="utf-8"))
    phase3 = phase3_raw[0] if isinstance(phase3_raw, list) and phase3_raw else phase3_raw
    shape_result = (phase3.get("shapes") or phase3.get("results") or [{}])[0]
    top_results = phase3.get("top_results") or []
    winner = (
        shape_result.get("winner_gflops")
        or nested(shape_result, "winner", "gflops")
        or phase3.get("winner_gflops")
        or (top_results[0].get("gflops") if top_results else None)
    )
    rows.append(
        {
            "variant": "minus_feedback_optimization",
            "provenance": "derived_from_shared_full_phase1_checkpoint",
            **full_base,
            "final_gflops": winner,
            "source": str(phase3_path),
        }
    )

    order = {
        "full": 0,
        "minus_state_in_prompt": 1,
        "intent_only_control": 2,
        "minus_local_checks": 3,
        "minus_stage_ordering": 4,
        "minus_feedback_optimization": 5,
        "minus_parameter_search": 6,
        "full_parameter_search_r4": 7,
    }
    rows.sort(key=lambda row: order[row["variant"]])
    fields = list(rows[0])
    with (output_dir / "rq2_complete_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "run_id": args.run_id,
        "shape": "512x512x512",
        "rows": rows,
        "warnings": [
            "Full and minus_parameter_search reuse an earlier archived run, as requested.",
            "Fresh construction ablations were executed after the ablation instrumentation changes; compare historical Full with care.",
            "Intent-only is the aggregate of three interface-repaired independent runs; failed runs are not encoded as zero throughput.",
            "The historical Intent-only 0/0 record is retained only in intent_only_repaired_summary.json as interface-failure provenance.",
            "minus_feedback_optimization starts from the archived Full Phase-1 code and was re-benchmarked in the current environment.",
        ],
    }
    (output_dir / "rq2_complete_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
