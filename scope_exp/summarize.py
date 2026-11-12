

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import HARNESS_VERSION, stats
from .paths import CampaignPaths
from .util import (ensure_dir, read_json, sha256_file, utc_now, write_csv, write_json)

SUMMARY_VERSION = "scope-exp-summary-v1"

BLOCK_LABELS = {
    "E1": "组件与阶段消融",
    "E2": "模型重复",
    "E3": "预算效率（扩展）",
    "D1": "依赖修订（扩展）",
    "PILOT": "先导",
}


def task_rows(paths: CampaignPaths, *, include_rehearsal: bool = False,
              protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:

    records = ((protocol or {}).get("tasks") or {}).get("task_records") or []
    rows: list[dict[str, Any]] = []
    if records:
        for record in records:
            lookup = record.get("reuse_of") or record["task_id"]
            status_path = paths.task_status(lookup)
            status = read_json(status_path) if status_path.exists() else {}
            if status.get("provenance") == "rehearsal" and not include_rehearsal:
                continue
            rows.append(_row_from_status(
                status, task_id=record["task_id"], lookup_id=lookup,
                block=record.get("block"), kind=record.get("kind"),
                shape=record.get("shape_M"), variant=record.get("variant"),
                model_role=record.get("model_role"),
                parent_task_id=record.get("parent_task_id"),
                reuse_of=record.get("reuse_of")))
        return rows

    for status_path in sorted(Path(paths.tasks).glob("*/status.json")):
        status = read_json(status_path)
        if status.get("provenance") == "rehearsal" and not include_rehearsal:
            continue
        task_id = status.get("task_id") or status_path.parent.name
        rows.append(_row_from_status(
            status, task_id=task_id, lookup_id=task_id,
            block=status.get("block"), kind=status.get("kind"),
            shape=status.get("shape"), variant=_variant_of(task_id),
            model_role=_model_of(task_id), parent_task_id=None,
            reuse_of=_reuse_of(task_id)))
    return rows


def _row_from_status(status: dict[str, Any], *, task_id: str, lookup_id: str,
                     block: Any, kind: Any, shape: Any, variant: Any,
                     model_role: Any, parent_task_id: Any,
                     reuse_of: Any) -> dict[str, Any]:
    validation = status.get("validation") or {}
    timing_summary = status.get("timing") or {}
    sanitizer_record = status.get("sanitizer") or {}
    outcome = status.get("outcome") or {}
    main = validation.get("main") or {}
    coefficients = validation.get("coefficients") or {}
    delivered = bool(outcome.get("delivered") or status.get("delivered_phase")
                or status.get("status") == "resolved")
    validation_pass = validation.get("verdict") == "pass"
    safety_status = sanitizer_record.get("status")
    complete = bool(delivered and validation_pass and
                    safety_status in ("pass", "pass_scoped", "not_applicable"))
    cost = status.get("cost") or {}
    tokens = cost.get("tokens_known") or {}
    return {
        "task_id": task_id,
        "evaluated_task_id": lookup_id,
        "shares_result_with": (lookup_id if lookup_id != task_id else None),
        "block": block,
        "kind": kind,
        "shape": shape,
        "status": status.get("status", "planned"),
        "provenance": status.get("provenance"),
        "variant": variant,
        "model_role": model_role,
        "parent_task_id": parent_task_id,
        "reuse_of": reuse_of,
        "delivered": delivered,
        "validation_pass": validation_pass,
        "validation_verdict": validation.get("verdict"),
        "main_passed": main.get("passed"),
        "main_planned": main.get("planned"),
        "main_failed": main.get("failed"),
        "main_not_run": main.get("not_run"),
        "coefficient_passed": coefficients.get("passed"),
        "coefficient_planned": coefficients.get("planned"),
        "strict_abs_passes": validation.get("strict_abs_total_pass"),
        "error_taxonomy": main.get("error_taxonomy"),
        "safety_status": safety_status,
        "safety_scope": sanitizer_record.get("scope"),
        "complete_delivery": complete,
        "gflops_mean": timing_summary.get("gflops_mean"),
        "gflops_median": timing_summary.get("gflops_median"),
        "gflops_sd": timing_summary.get("gflops_sd"),
        "latency_ms_mean": timing_summary.get("mean_latency_ms"),
        "timing_sessions": timing_summary.get("n_sessions"),
        "validation_binary_sha256": validation.get("binary_sha256"),
        "timing_binary_sha256": timing_summary.get("binary_sha256"),
        "config_hash": status.get("resolved_config_hash"),
        "delivered_phase": status.get("delivered_phase"),
        "attempts": len(status.get("attempts") or []),
        "infra_errors": len(status.get("infra_errors") or []),
        "requests": cost.get("requests"),
        "cost_usd": cost.get("cost_usd_known"),
        "cost_complete": cost.get("cost_is_complete"),
        "prompt_tokens": tokens.get("prompt_tokens"),
        "completion_tokens": tokens.get("completion_tokens"),
        "outcome_category": outcome.get("category"),

        "diagnosis": status.get("generation_diagnosis") or outcome.get("diagnosis"),
        "updated_at": status.get("updated_at"),
    }


def _reuse_of(task_id: str) -> str | None:
    return task_id if task_id.startswith("E1-full-") else None


def _variant_of(task_id: str) -> str | None:
    parts = task_id.split("-")
    if len(parts) >= 3 and parts[0] in ("E1", "E2", "E3", "D1"):
        if parts[0] == "E1":
            return ("minus_parameter_search" if parts[1] == "woP3" else
                    "minus_feedback_optimization" if parts[1] == "woFB" else parts[1])
        if parts[0] == "E2":
            return "full"
        if parts[0] in ("D1", "E3"):
            return parts[1]
    return None


def _model_of(task_id: str) -> str | None:
    parts = task_id.split("-")
    if len(parts) >= 4 and parts[0] == "E2":
        return "-".join(parts[1:-2])
    return None


def _describe(values: Iterable[float | None], *, alpha: float, resamples: int,
              seed: int) -> dict[str, Any]:
    return stats.describe([value for value in values if value is not None],
                          alpha=alpha, bootstrap_resamples=resamples, seed=seed)


def group_table(rows: Sequence[dict[str, Any]], group_keys: Sequence[str],
                *, alpha: float = 0.05, bootstrap_resamples: int = 10000,
                bootstrap_seed: int = 20261008) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(name) for name in group_keys), []).append(row)
    table: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        record = {name: value for name, value in zip(group_keys, key)}
        planned = len(items)
        started = sum(1 for item in items if item["status"] != "planned")
        resolved = sum(1 for item in items if item["status"] == "resolved")
        infra_unknown = sum(1 for item in items if item["status"] == "infra_incomplete")
        complete = sum(1 for item in items if item["complete_delivery"])
        science_failure = sum(
            1 for item in items
            if item["status"] in ("generation_failed", "no_candidate") or
            (item["status"] == "resolved" and not item["delivered"]))
        throughput = _describe([item["gflops_mean"] for item in items if item["delivered"]],
                               alpha=alpha, resamples=bootstrap_resamples,
                               seed=bootstrap_seed)
        rates = stats.rate_summary(complete, planned, unresolved=infra_unknown, alpha=alpha)
        record.update({
            "n_planned": planned,
            "n_started": started,
            "n_resolved": resolved,
            "n_infra_unknown": infra_unknown,
            "n_generated": sum(1 for item in items if item["delivered"]),
            "n_validated": sum(1 for item in items if item["validation_pass"]),
            "n_complete": complete,
            "n_science_failure": science_failure,
            "rate_complete": rates["rate"],
            "rate_complete_lower": rates["wilson_lower"],
            "rate_complete_upper": rates["wilson_upper"],
            "rate_complete_range": rates["complete_queue_range"],
            "complete_queue_range": rates["complete_queue_range"],
            "rate_interval_scope": rates["interval_scope"],
            "throughput": throughput,
            "throughput_mean": throughput["mean"],
            "throughput_sd": throughput["sd"],
            "throughput_median": throughput["median"],
            "throughput_cv": throughput["cv"],
            "throughput_n": throughput["n"],
            "throughput_ci_low": throughput["ci_low"],
            "throughput_ci_high": throughput["ci_high"],
            "throughput_note": throughput["note"],
            "effective_throughput": stats.effective_throughput(
                [item["delivered"] for item in items],
                [item["gflops_mean"] for item in items]),
            "requests": sum(item["requests"] or 0 for item in items),
            "cost_usd_known": sum(item["cost_usd"] or 0.0 for item in items),
            "cost_complete": all(bool(item["cost_complete"]) for item in items) if items else False,
        })
        record["effective_throughput_mean"] = record["effective_throughput"]["mean"]
        table.append(record)
    return table


def stage_pairs(rows: Sequence[dict[str, Any]], *, alpha: float = 0.05,
                bootstrap_resamples: int = 10000,
                bootstrap_seed: int = 20261008) -> list[dict[str, Any]]:

    by_id = {row["task_id"]: row for row in rows}
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        parts = row["task_id"].split("-")
        if len(parts) < 4 or parts[0] != "E1" or parts[1] not in ("woP3", "woFB"):
            continue
        parent_id = f"E1-full-{parts[2]}-{parts[3]}"
        kind = ("without_parameter_search" if parts[1] == "woP3"
                else "without_feedback")
        pairs.setdefault((kind, str(row["shape"])), []).append(
            {"parent": by_id.get(parent_id), "branch": row})
    table: list[dict[str, Any]] = []
    for (kind, shape), entries in sorted(pairs.items()):
        throughput_differences: list[float] = []
        delivery_differences: list[float] = []
        resolved_pairs = 0
        for entry in entries:
            parent, branch = entry["parent"], entry["branch"]
            if parent is None or parent["status"] == "infra_incomplete" or \
                    branch["status"] == "infra_incomplete":
                continue
            resolved_pairs += 1
            delivery_differences.append(float(bool(branch["delivered"])) -
                                        float(bool(parent["delivered"])))
            if parent["gflops_mean"] is not None and branch["gflops_mean"] is not None:
                throughput_differences.append(float(branch["gflops_mean"]) -
                                              float(parent["gflops_mean"]))
        paired_throughput = stats.paired_bootstrap(
            throughput_differences, resamples=bootstrap_resamples,
            seed=bootstrap_seed, alpha=alpha)
        paired_delivery = stats.paired_bootstrap(
            delivery_differences, resamples=bootstrap_resamples,
            seed=bootstrap_seed, alpha=alpha)
        table.append({
            "stage": kind,
            "shape": shape,
            "n_pairs_planned": len(entries),
            "n_pairs_resolved": resolved_pairs,
            "n_pairs_both_delivered": len(throughput_differences),
            "mean_throughput_difference": paired_throughput["mean_difference"],
            "throughput_difference_ci_low": paired_throughput["ci_low"],
            "throughput_difference_ci_high": paired_throughput["ci_high"],
            "mean_delivery_difference": paired_delivery["mean_difference"],
            "delivery_difference_ci_low": paired_delivery["ci_low"],
            "delivery_difference_ci_high": paired_delivery["ci_high"],
            "paired": True,
            "independent_n": 0,
            "note": "difference = branch - parent; resampled as pairs so the shared "
                    "prefix is not counted twice",
        })
    return table


def summarize_campaign(paths: CampaignPaths, *, include_rehearsal: bool = False,
                       alpha: float = 0.05, bootstrap_resamples: int = 10000,
                       bootstrap_seed: int = 20261008,
                       protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = task_rows(paths, include_rehearsal=include_rehearsal, protocol=protocol)
    independent = [row for row in rows if row["kind"] in (None, "independent")]
    component_rows = [row for row in rows
                      if row["block"] == "E1" and row["kind"] == "independent"]
    model_rows = [row for row in rows
                  if row["block"] == "E2" and row["kind"] in ("independent", "alias")]
    summary: dict[str, Any] = {
        "summary_version": SUMMARY_VERSION,
        "harness_version": HARNESS_VERSION,
        "created_at": utc_now(),
        "campaign_id": paths.campaign_id,
        "include_rehearsal": include_rehearsal,
        "counts": {
            "rows": len(rows),
            "independent": len(independent),
            "branches": sum(1 for row in rows if row["kind"] == "branch"),
            "aliases": sum(1 for row in rows if row["kind"] == "alias"),
            "by_status": _count_by(rows, "status"),
            "by_provenance": _count_by(rows, "provenance"),
            "by_block": _count_by(rows, "block"),
        },
        "tables": {
            "rq2_components": group_table(component_rows, ["variant", "shape"],
                                          alpha=alpha,
                                          bootstrap_resamples=bootstrap_resamples,
                                          bootstrap_seed=bootstrap_seed),
            "rq2_components_pooled": group_table(component_rows, ["variant"],
                                                 alpha=alpha,
                                                 bootstrap_resamples=bootstrap_resamples,
                                                 bootstrap_seed=bootstrap_seed),
            "models": group_table(model_rows, ["model_role", "shape"], alpha=alpha,
                                  bootstrap_resamples=bootstrap_resamples,
                                  bootstrap_seed=bootstrap_seed),
            "models_pooled": group_table(model_rows, ["model_role"], alpha=alpha,
                                         bootstrap_resamples=bootstrap_resamples,
                                         bootstrap_seed=bootstrap_seed),
            "stage_pairs": stage_pairs(rows, alpha=alpha,
                                       bootstrap_resamples=bootstrap_resamples,
                                       bootstrap_seed=bootstrap_seed),
            "correctness": correctness_table(rows),
            "cost": cost_table(rows),
            "completeness": completeness_table(rows, protocol=protocol),
        },
        "caveats": [
            "a rate is s/N with a Wilson interval; with u unresolved tasks the "
            "complete-queue range [s/N, (s+u)/N] is reported instead of hiding them",
            "n=0 reports no mean; n=1 reports no estimable SD or interval",
            "input-level checks (64 per kernel per shape) are correlated and are never "
            "used as independent samples",
            "an insignificant difference is not equivalence",
            "Y_i = delivered_i * throughput_i is a rate per generation attempt, not a "
            "kernel running speed",
            "an E2 row that reuses an E1 Full slot (shares_result_with) is the same "
            "task: never sum the E1 and E2 columns to get an independent N",
            "rehearsal rows are excluded unless include_rehearsal is set",
        ] + ([
            "this is a CPU campaign: the frozen method disables the feedback stage "
            "(P2) when the backend is cpu, so no without_feedback comparison exists "
            "and none is reported",
        ] if str((protocol or {}).get("backend", "")) == "cpu" else []),
    }
    return summary


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def correctness_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["task_id"]):
        table.append({
            "task_id": row["task_id"],
            "block": row["block"],
            "kind": row["kind"],
            "shape": row["shape"],
            "variant": row["variant"],
            "status": row["status"],
            "validation_verdict": row["validation_verdict"],
            "main_passed": row["main_passed"],
            "main_planned": row["main_planned"],
            "coefficient_passed": row["coefficient_passed"],
            "coefficient_planned": row["coefficient_planned"],
            "strict_abs_passes": row["strict_abs_passes"],
            "safety_status": row["safety_status"],
            "safety_scope": row["safety_scope"],
            "outcome_category": row["outcome_category"],
            "diagnosis_reason": (row.get("diagnosis") or {}).get("reason"),
            "diagnosis_chains": (row.get("diagnosis") or {}).get("chain_count"),
            "diagnosis_accepted": (row.get("diagnosis") or {}).get("accepted_count"),
            "diagnosis_compile_pass": (row.get("diagnosis") or {}).get("compile_pass_count"),
            "diagnosis_correctness_pass": (row.get("diagnosis") or {}).get(
                "correctness_pass_count"),
            "complete_delivery": row["complete_delivery"],
            "error_taxonomy": row["error_taxonomy"],
        })
    return table


def cost_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"task_id": row["task_id"], "block": row["block"],
             "variant": row["variant"], "model_role": row["model_role"],
             "requests": row["requests"], "prompt_tokens": row["prompt_tokens"],
             "completion_tokens": row["completion_tokens"],
             "cost_usd_known": row["cost_usd"],
             "cost_complete": row["cost_complete"],
             "attempts": row["attempts"], "infra_errors": row["infra_errors"]}
            for row in sorted(rows, key=lambda item: item["task_id"])]


def completeness_table(rows: Sequence[dict[str, Any]],
                       *, protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    planned_counts = (((protocol or {}).get("tasks") or {}).get("counts") or {})
    table: list[dict[str, Any]] = []
    by_block: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(str(row["block"]), []).append(row)
    for block, items in sorted(by_block.items()):
        statuses = _count_by(items, "status")
        table.append({
            "scope": "block", "name": block, "n_planned": len(items),
            "n_resolved": statuses.get("resolved", 0),
            "n_infra_incomplete": statuses.get("infra_incomplete", 0),
            "n_no_candidate": statuses.get("no_candidate", 0),
            "n_generation_failed": statuses.get("generation_failed", 0),
            "n_planned_not_started": statuses.get("planned", 0),
            "protocol_expected": planned_counts.get(block),
        })
    statuses = _count_by(rows, "status")
    table.append({
        "scope": "campaign", "name": "all", "n_planned": len(rows),
        "n_resolved": statuses.get("resolved", 0),
        "n_infra_incomplete": statuses.get("infra_incomplete", 0),
        "n_no_candidate": statuses.get("no_candidate", 0),
        "n_generation_failed": statuses.get("generation_failed", 0),
        "n_planned_not_started": statuses.get("planned", 0),
        "protocol_expected": planned_counts.get("core_independent_total"),
    })
    return table


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "--"
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 1e-3:
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def _rate_text(record: dict[str, Any]) -> str:
    text = (f'{record.get("n_complete", 0)}/{record.get("n_planned")}'
            if record.get("n_planned") else "--")
    low, high = record.get("rate_complete_lower"), record.get("rate_complete_upper")
    if low is not None and high is not None:
        text += f" (Wilson {low:.3f}–{high:.3f})"
    return text


def _throughput_text(record: dict[str, Any]) -> str:
    mean = record.get("throughput_mean")
    if mean is None:
        return "--"
    text = f"{mean:.1f}"
    sd = record.get("throughput_sd")
    if sd is not None:
        text += f" ± {sd:.1f}"
    low, high = record.get("throughput_ci_low"), record.get("throughput_ci_high")
    if low is not None and high is not None:
        text += f" [{low:.1f}, {high:.1f}]"
    return text + f" (n={record.get('throughput_n')})"


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    counts = summary["counts"]
    lines.append(f"# 结果汇总（campaign `{summary['campaign_id']}`）")
    lines.append("")
    lines.append(f"生成时间：{summary['created_at']}；harness {summary['harness_version']}。")
    lines.append("")
    lines.append("## 0. 完整度")
    lines.append("")
    lines.append("| 范围 | 计划 | 已解决 | 基础设施未决 | 无候选 | 生成失败 | 未开始 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["tables"]["completeness"]:
        lines.append(f'| {row["name"]} | {row["n_planned"]} | {row["n_resolved"]} | '
                     f'{row["n_infra_incomplete"]} | {row["n_no_candidate"]} | '
                     f'{row["n_generation_failed"]} | {row["n_planned_not_started"]} |')
    lines.append("")
    lines.append(f"任务状态分布：`{counts['by_status']}`；来源分布：`{counts['by_provenance']}`。")
    lines.append("")
    lines.append("## 1. RQ2 组件消融（按形状分别报告）")
    lines.append("")
    lines.append("| 配置 | 形状 | 完整交付 | 合格吞吐 GFLOP/s | 请求数 |")
    lines.append("|---|---:|---|---|---:|")
    for row in summary["tables"]["rq2_components"]:
        lines.append(f'| {row["variant"]} | {row["shape"]} | {_rate_text(row)} | '
                     f'{_throughput_text(row)} | {row["requests"]} |')
    lines.append("")
    lines.append("跨形状汇总仅作补充，不掩盖单个形状上的失败或收益反转。")
    lines.append("")
    lines.append("| 配置 | 完整交付 | 合格吞吐 |")
    lines.append("|---|---|---|")
    for row in summary["tables"]["rq2_components_pooled"]:
        lines.append(f'| {row["variant"]} | {_rate_text(row)} | {_throughput_text(row)} |')
    lines.append("")
    lines.append("## 2. 模型重复")
    lines.append("")
    lines.append("| 模型 | 形状 | 完整交付 | 合格吞吐 |")
    lines.append("|---|---:|---|---|")
    for row in summary["tables"]["models"]:
        lines.append(f'| {row["model_role"]} | {row["shape"]} | {_rate_text(row)} | '
                     f'{_throughput_text(row)} |')
    lines.append("")
    lines.append("## 3. 阶段配对（共享父任务）")
    lines.append("")
    lines.append("| 阶段 | 形状 | 配对数 | 吞吐差 mean [CI] | 交付率差 mean [CI] |")
    lines.append("|---|---:|---:|---|---|")
    for row in summary["tables"]["stage_pairs"]:
        throughput = (f'{_fmt(row["mean_throughput_difference"], digits=2)} '
                      f'[{_fmt(row["throughput_difference_ci_low"], digits=2)}, '
                      f'{_fmt(row["throughput_difference_ci_high"], digits=2)}]')
        delivery = (f'{_fmt(row["mean_delivery_difference"])} '
                    f'[{_fmt(row["delivery_difference_ci_low"])}, '
                    f'{_fmt(row["delivery_difference_ci_high"])}]')
        lines.append(f'| {row["stage"]} | {row["shape"]} | {row["n_pairs_resolved"]} | '
                     f'{throughput} | {delivery} |')
    lines.append("")
    lines.append("差值为「分支 − 父任务」，按父任务成对重采样；共享前缀不重复计入成本。")
    lines.append("")
    lines.append("## 4. 正确性")
    lines.append("")
    lines.append("| 任务 | 状态 | 原因 | 复验 | 主检查 | 系数检查 | 严格 abs | 安全 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in summary["tables"]["correctness"]:
        main = (f'{row["main_passed"]}/{row["main_planned"]}'
                if row["main_planned"] else "--")
        coefficient = (f'{row["coefficient_passed"]}/{row["coefficient_planned"]}'
                       if row["coefficient_planned"] else "--")
        strict = row["strict_abs_passes"] if row["strict_abs_passes"] is not None else "--"
        lines.append(f'| {row["task_id"]} | {row["status"]} | '
                     f'{_fmt(row["outcome_category"])} | '
                     f'{_fmt(row["validation_verdict"])} | {main} | {coefficient} | '
                     f'{strict} | {_fmt(row["safety_status"])} |')
    lines.append("")
    diagnoses = [row for row in summary["tables"]["correctness"]
                 if row.get("diagnosis_reason")]
    if diagnoses:
        lines.append("### 4.1 未交付任务的自我诊断")
        lines.append("")
        lines.append("| 任务 | 原因 | 终态链 | 被接受 | 编译通过 | 正确性通过 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in diagnoses:
            lines.append(f'| {row["task_id"]} | {row["diagnosis_reason"]} | '
                         f'{_fmt(row.get("diagnosis_chains"))} | '
                         f'{_fmt(row.get("diagnosis_accepted"))} | '
                         f'{_fmt(row.get("diagnosis_compile_pass"))} | '
                         f'{_fmt(row.get("diagnosis_correctness_pass"))} |')
        lines.append("")
        lines.append("诊断来自方法自己的判据文件（`app_evidence/`），不是平台的猜测。")
        lines.append("")
    lines.append("## 5. 成本")
    lines.append("")
    lines.append("| 任务 | 请求 | prompt tokens | completion tokens | 费用(USD) | 费用完整 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in summary["tables"]["cost"]:
        lines.append(f'| {row["task_id"]} | {_fmt(row["requests"])} | '
                     f'{_fmt(row["prompt_tokens"])} | {_fmt(row["completion_tokens"])} | '
                     f'{_fmt(row["cost_usd_known"])} | {_fmt(row["cost_complete"])} |')
    lines.append("")
    lines.append("未知金额保持空值，不填 0；共享前缀只计一次。")
    lines.append("")
    lines.append("## 6. 报告规则")
    lines.append("")
    for caveat in summary["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def write_reports(paths: CampaignPaths, summary: dict[str, Any]) -> dict[str, Path]:
    ensure_dir(paths.analysis)
    outputs: dict[str, Path] = {}
    for name, table in summary["tables"].items():
        if not table:
            continue
        outputs[f"{name}.csv"] = write_csv(paths.analysis / f"{name}.csv", table)
    outputs["summary.json"] = write_json(paths.analysis / "summary.json", summary)
    results_path = paths.analysis / "RESULTS_zh.md"
    results_path.write_text(render_markdown(summary), encoding="utf-8")
    outputs["RESULTS_zh.md"] = results_path
    return outputs


def figures(paths: CampaignPaths, summary: dict[str, Any]) -> dict[str, Any]:

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}"}
    ensure_dir(paths.figures)
    written: list[str] = []
    components = summary["tables"]["rq2_components"]
    if components:
        shapes = sorted({row["shape"] for row in components})
        variants = sorted({row["variant"] for row in components})
        figure, axis = plt.subplots(figsize=(8, 4.5))
        width = 0.8 / max(len(variants), 1)
        for index, variant in enumerate(variants):
            values = [next((row["throughput_mean"] or 0.0 for row in components
                            if row["variant"] == variant and row["shape"] == shape), 0.0)
                      for shape in shapes]
            axis.bar([position + index * width for position in range(len(shapes))],
                     values, width, label=variant)
        axis.set_xticks([position + width * (len(variants) - 1) / 2
                         for position in range(len(shapes))])
        axis.set_xticklabels([str(shape) for shape in shapes])
        axis.set_ylabel("delivered throughput (GFLOP/s)")
        axis.set_xlabel("shape")
        axis.legend(fontsize=6)
        path = paths.figures / "rq2_components_throughput.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(str(path))
    return {"status": "written", "files": written}


def analysis_status(paths: CampaignPaths) -> dict[str, Any]:
    path = paths.analysis / "summary.json"
    if not path.exists():
        return {"present": False, "path": str(path)}
    record = read_json(path)
    return {"present": True, "created_at": record.get("created_at"),
            "counts": record.get("counts"), "path": str(path),
            "sha256": sha256_file(path)}
