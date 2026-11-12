

from __future__ import annotations

import math
from typing import Any, Sequence

from . import HARNESS_VERSION, runner, stats, tasks as tasks_module
from .paths import CampaignPaths
from .util import ensure_dir, read_json, sha256_value, utc_now, write_csv, write_json

PILOT_VERSION = "scope-exp-pilot-v1"


def pilot_protocol(protocol: dict[str, Any]) -> dict[str, Any]:

    shapes = [int(shape) for shape in (protocol.get("shapes") or tasks_module.SHAPES)]
    model_roles = [role for role in (protocol.get("models") or {})
                   if role != "rehearsal"] or list(tasks_module.MODEL_ROLES)
    pilot_tasks = tasks_module.pilot_tasks(shapes=shapes, model_roles=model_roles)
    matrix = tasks_module.validate_matrix(pilot_tasks, blocks=("PILOT",),
                                          shapes=shapes,
                                          model_roles=model_roles)
    clone = dict(protocol)
    clone["tasks"] = {
        "counts": matrix["observed"],
        "expected": matrix["expected"],
        "task_ids": [task.task_id for task in pilot_tasks],
        "order": [task.task_id for task in sorted(pilot_tasks,
                                                 key=lambda item: item.order_index)],
        "manifest_sha256": sha256_value([task.task_id for task in pilot_tasks]),
        "blocks": ["PILOT"],
        "task_records": [task.as_record() for task in pilot_tasks],
        "denominators": {"independent": len(pilot_tasks), "branches": 0, "aliases": 0},
    }
    clone["pilot_of"] = protocol.get("protocol_hash")
    clone["created_at"] = utc_now()
    return clone


def plan(paths: CampaignPaths, protocol: dict[str, Any]) -> dict[str, Any]:
    clone = pilot_protocol(protocol)
    record = {
        "pilot_version": PILOT_VERSION,
        "harness_version": HARNESS_VERSION,
        "created_at": utc_now(),
        "campaign_id": paths.campaign_id,
        "tasks": clone["tasks"]["task_records"],
        "counts": clone["tasks"]["counts"],
        "budget": {
            "per_task_wall_seconds": (protocol.get("budget") or {}).get("pilot_wall_seconds"),
            "split": "development",
            "note": "pilot data never enters the formal statistics",
        },
        "excluded_from_formal": True,
    }
    ensure_dir(paths.analysis)
    write_json(paths.analysis / "pilot_plan.json", record)
    return record


def run(paths: CampaignPaths, protocol: dict[str, Any], options: runner.RunOptions, *,
        task_ids: Sequence[str] | None = None,
        stop_on_infra_error: bool = True) -> dict[str, Any]:
    clone = pilot_protocol(protocol)
    options = runner.RunOptions(
        campaign_id=options.campaign_id, backend=options.backend,
        generator=options.generator, mode=options.mode, device=options.device,
        keep_workspace=options.keep_workspace, skip_sanitizer=options.skip_sanitizer,
        allow_contended=options.allow_contended, dry_run=options.dry_run,
        evaluate=options.evaluate, time_kernel=options.time_kernel,
        provenance="pilot", pricing=options.pricing,
        lock_timeout_seconds=options.lock_timeout_seconds,
        isolate=options.isolate)
    summary = runner.run_campaign(paths, clone, options, task_ids=task_ids,
                                  stop_on_infra_error=stop_on_infra_error)
    report = build_report(paths, clone)
    return {"run": summary, "report": report}


def build_report(paths: CampaignPaths, protocol: dict[str, Any]) -> dict[str, Any]:
    rows = runner.task_table(paths, protocol)
    durations: list[float] = []

    phases: dict[str, int] = {"p1": 0, "p2": 0, "p3": 0}
    categories: dict[str, int] = {}
    requests = 0
    costs_unknown = 0
    for row in rows:
        status = runner.read_status(paths, row["task_id"])
        for attempt in status.get("attempts") or []:
            if attempt.get("wall_seconds"):
                durations.append(float(attempt["wall_seconds"]))
        if row["outcome_category"]:
            categories[row["outcome_category"]] = categories.get(row["outcome_category"], 0) + 1
        requests += row["cost_requests"] or 0
        if row["provenance"] == "pilot" and not row["cost_complete"]:
            costs_unknown += 1
        for phase in ("p1", "p2", "p3"):
            if (status.get("checkpoints") or {}).get(phase):
                phases[phase] += 1
    delivered = sum(1 for row in rows if row["delivered"])
    resolved = sum(1 for row in rows if row["status"] == "resolved")
    infra = sum(1 for row in rows if row["status"] == "infra_incomplete")
    validated = sum(1 for row in rows if row["validation_verdict"] == "pass")
    timed = sum(1 for row in rows if row["gflops_mean"] is not None)
    planned = len(rows)
    wall = stats.describe(durations) if durations else stats.describe([])
    report = {
        "pilot_version": PILOT_VERSION,
        "created_at": utc_now(),
        "tasks": len(rows),
        "resolved": resolved,
        "delivered": delivered,
        "infrastructure_incomplete": infra,
        "failure_taxonomy": dict(sorted(categories.items())),
        "wall_seconds": wall,
        "phase_checkpoint_counts": phases,
        "requests": requests,
        "cost_complete_tasks": sum(1 for row in rows if row["cost_complete"]),
        "cost_unknown_tasks": costs_unknown,
        "validations_passed": validated,
        "timing_measured": timed,
        "tasks": [
            {"task_id": row["task_id"], "status": row["status"],
             "delivered": row["delivered"],
             "validation_verdict": row["validation_verdict"],
             "gflops_mean": row["gflops_mean"],
             "outcome_category": row["outcome_category"],
             "cost_requests": row["cost_requests"]}
            for row in rows
        ],
        "sample_size": sample_size_recommendation(delivered, planned),
        "budget_recommendation": budget_recommendation(wall),
        "excluded_from_formal": True,
        "verdict": _verdict(infra, delivered),
        "notes": [
            "pilot rows are labelled provenance=pilot and stay in the development split",
            "a pilot that is cut short by a wall-clock limit is recorded as truncated, "
            "not as a complete default search",
        ],
    }
    ensure_dir(paths.analysis)
    write_json(paths.analysis / "pilot_report.json", report)
    write_csv(paths.analysis / "pilot_tasks.csv", rows)
    return report


def _verdict(infrastructure_incomplete: int, delivered: int) -> str:
    if infrastructure_incomplete:
        return "infrastructure_blocked"
    if delivered == 0:
        return "no_delivery_observed"
    return "usable"


def sample_size_recommendation(successes: int, total: int, *,
                               target_half_width: float = 0.10,
                               alpha: float = 0.05) -> dict[str, Any]:

    observed = (successes / total) if total else None
    z = 1.959963984540054
    def needed(p: float) -> int:
        return int(math.ceil((z * z * p * (1.0 - p)) / (target_half_width ** 2)))
    worst_case = needed(0.5)
    at_observed = needed(observed) if observed is not None else None
    table = []
    for n in (10, 20, 30):
        interval = stats.wilson_interval(n, n, alpha=alpha) if n else None
        table.append({"n": n, "all_pass_lower_bound": interval[0] if interval else None})
    return {
        "observed_rate_in_pilot": observed,
        "target_half_width": target_half_width,
        "required_n_worst_case": worst_case,
        "required_n_at_observed_rate": at_observed,
        "all_pass_lower_bounds": table,
        "recommendation": ("keep 10 repeats if the pilot spread is small and the effect "
                           "size is large; otherwise raise to 20 or 30 before freezing. "
                           "The decision is made after the pilot and before any formal "
                           "result exists, and is then frozen."
                           if observed is not None else
                           "insufficient pilot data to recommend a sample size"),
        "caveat": "10/10 success gives a Wilson 95% lower bound of about 0.722; it is "
                  "not evidence of 99% reliability",
    }


def budget_recommendation(wall: dict[str, Any]) -> dict[str, Any]:
    mean = wall.get("mean")
    high = wall.get("max")
    if mean is None:
        return {"status": "no_data",
                "note": "no completed pilot attempt; keep the 8h debug guard and do "
                        "not freeze a formal budget yet"}
    return {
        "status": "measured",
        "mean_wall_seconds": mean,
        "max_wall_seconds": high,
        "suggested_per_task_wall_seconds": round(max(high * 1.5, mean * 2.0), 1),
        "suggested_pilot_wall_seconds": round(max(high * 1.5, mean * 2.0) * 2, 1),
        "scenario_280_tasks_hours": {
            "at_pilot_mean": round(280 * mean / 3600.0, 1),
            "at_pilot_max": round(280 * (high or mean) / 3600.0, 1),
        },
        "note": "a suggestion, not a forecast; the frozen value must be decided "
                "before the formal queue starts",
    }
