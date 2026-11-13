
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .util import (append_jsonl, ensure_dir, read_json, read_jsonl, utc_now,
                   write_csv)

LEDGER_VERSION = "scope-exp-ledger-v1"
LEDGER_ENV_VAR = "SCOPE_EXP_LEDGER"
PHASE_ENV_VAR = "SCOPE_EXP_LEDGER_PHASE"

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens",
                "cached_tokens", "reasoning_tokens")


def ledger_path_for(attempt_dir: Path) -> Path:
    return Path(attempt_dir) / "api_ledger.jsonl"


def record_request(*, request_id: str | None, task_id: str, attempt_id: str,
                   phase: str | None, call_type: str, started_at: str,
                   duration_seconds: float | None, failed: bool,
                   retry_of: str | None = None, tokens: dict[str, Any] | None = None,
                   usage_source: str = "instrumented_client",
                   returned_model: str | None = None,
                   error: str | None = None) -> dict[str, Any]:
    token_values: dict[str, int | None] = {field: None for field in TOKEN_FIELDS}
    for field in TOKEN_FIELDS:
        if tokens and field in tokens:
            value = tokens[field]
            token_values[field] = int(value) if isinstance(value, (int, float)) else None
    return {
        "ledger_version": LEDGER_VERSION,
        "request_id": request_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "phase": phase,
        "call_type": call_type,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "failed": bool(failed),
        "retry_of": retry_of,
        "tokens": token_values,
        "usage_source": usage_source,
        "returned_model": returned_model,
        "error": error,
        "cost_usd": None,
        "cost_source": "unknown",
    }


class ApiLedger:

    def __init__(self, path: Path, *, task_id: str, attempt_id: str,
                 phase: str | None = None) -> None:
        self.path = Path(path)
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.phase = phase
        ensure_dir(self.path.parent)

    def write(self, row: dict[str, Any]) -> dict[str, Any]:
        append_jsonl(self.path, row)
        return row

    def read(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path)

    def add(self, *, call_type: str, duration_seconds: float | None,
            failed: bool, request_id: str | None = None,
            tokens: dict[str, Any] | None = None, returned_model: str | None = None,
            retry_of: str | None = None, error: str | None = None,
            phase: str | None = None) -> dict[str, Any]:
        return self.write(record_request(
            request_id=request_id, task_id=self.task_id, attempt_id=self.attempt_id,
            phase=phase or self.phase, call_type=call_type, started_at=utc_now(),
            duration_seconds=duration_seconds, failed=failed, retry_of=retry_of,
            tokens=tokens, returned_model=returned_model, error=error))


def pricing_from_mapping(mapping: dict[str, Any] | None,
                         returned_model: str | None) -> dict[str, Any] | None:
    if not mapping or not returned_model:
        return None
    entry = mapping.get(returned_model)
    if not entry:
        return None
    return {"model": returned_model,
            "input_usd_per_mtok": entry.get("input"),
            "output_usd_per_mtok": entry.get("output"),
            "source": "operator_supplied"}


def cost_of(row: dict[str, Any], pricing: dict[str, Any] | None) -> dict[str, Any]:
    price = pricing_from_mapping(pricing, row.get("returned_model"))
    if price is None:
        return {"cost_usd": None, "cost_source": "unknown"}
    tokens = row.get("tokens") or {}
    prompt = tokens.get("prompt_tokens")
    completion = tokens.get("completion_tokens")
    if prompt is None or completion is None:
        return {"cost_usd": None, "cost_source": "unknown_token_count"}
    if price["input_usd_per_mtok"] is None or price["output_usd_per_mtok"] is None:
        return {"cost_usd": None, "cost_source": "incomplete_price"}
    value = (prompt / 1e6) * float(price["input_usd_per_mtok"]) + \
            (completion / 1e6) * float(price["output_usd_per_mtok"])
    return {"cost_usd": round(value, 8), "cost_source": f"price:{price['model']}"}


def summarize(rows: Sequence[dict[str, Any]], *,
              pricing: dict[str, Any] | None = None,
              phase_aggregates: dict[str, Any] | None = None) -> dict[str, Any]:
    total = {field: 0 for field in TOKEN_FIELDS}
    unknown_tokens = {field: 0 for field in TOKEN_FIELDS}
    cost = 0.0
    unknown_cost = 0
    failed = 0
    duration = 0.0
    per_model: dict[str, int] = {}
    per_call_type: dict[str, int] = {}
    for row in rows:
        if row.get("failed"):
            failed += 1
        duration += float(row.get("duration_seconds") or 0.0)
        tokens = row.get("tokens") or {}
        for field in TOKEN_FIELDS:
            value = tokens.get(field)
            if value is None:
                unknown_tokens[field] += 1
            else:
                total[field] += int(value)
        priced = cost_of(row, pricing)
        if priced["cost_usd"] is None:
            unknown_cost += 1
        else:
            cost += priced["cost_usd"]
        model = str(row.get("returned_model") or "unknown")
        per_model[model] = per_model.get(model, 0) + 1
        call_type = str(row.get("call_type") or "unknown")
        per_call_type[call_type] = per_call_type.get(call_type, 0) + 1
    return {
        "ledger_version": LEDGER_VERSION,
        "requests": len(rows),
        "failed_requests": failed,
        "request_seconds": round(duration, 6),
        "tokens_known": total,
        "tokens_unknown_rows": unknown_tokens,
        "cost_usd_known": round(cost, 8),
        "cost_unknown_rows": unknown_cost,
        "cost_is_complete": unknown_cost == 0,
        "cost_note": ("no request was made, so the cost is known to be zero"
                      if not rows else
                      "every request has a price and a token count"
                      if unknown_cost == 0 else
                      "some rows have no price/usage and stay unknown"),
        "per_model": per_model,
        "per_call_type": per_call_type,
        "phase_aggregates_app_snapshots": phase_aggregates or {},
        "phase_aggregates_note": "the frozen client does not expose a phase label per "
                                 "request; phase attribution comes from the "
                                 "application's own usage snapshots and is reported "
                                 "separately from the per-request rows",
        "unknown_is_not_zero": True,
    }


def rows_to_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    columns = ["request_id", "task_id", "attempt_id", "phase", "call_type",
               "started_at", "duration_seconds", "failed", "retry_of",
               "returned_model", "usage_source", "error"]
    flattened: list[dict[str, Any]] = []
    for row in rows:
        record = {column: row.get(column) for column in columns}
        tokens = row.get("tokens") or {}
        for field in TOKEN_FIELDS:
            record[field] = tokens.get(field)
        record["cost_usd"] = row.get("cost_usd")
        flattened.append(record)
    columns = columns + list(TOKEN_FIELDS) + ["cost_usd"]
    return write_csv(path, flattened, columns)


def collect_campaign_ledger(paths: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_task: dict[str, Any] = {}
    for status_path in sorted(Path(paths.tasks).glob("*/status.json")):
        task_id = status_path.parent.name
        task_rows: list[dict[str, Any]] = []
        for ledger_file in sorted(status_path.parent.glob("attempts/*/api_ledger.jsonl")):
            task_rows.extend(read_jsonl(ledger_file))
        rows.extend(task_rows)
        per_task[task_id] = {"requests": len(task_rows),
                             "failed": sum(1 for row in task_rows if row.get("failed"))}
    return {"summary": summarize(rows), "per_task": per_task,
            "shared_prefix_deduplication": {
                "rule": "a branch reusing a parent checkpoint contributes no new "
                        "requests of its own to the parent's prefix",
                "applied": True,
            }}


def app_phase_aggregates(rq3_path: Path) -> dict[str, Any]:
    if not Path(rq3_path).exists():
        return {}
    record = read_json(rq3_path)
    phase2 = (record.get("phase2_feedback_optimization") or {}).get("llm_usage") or {}
    return {
        "p2_feedback": phase2,
        "p3_parameter_search": record.get("phase3_constrained_reinstantiation", {}).get(
            "aggregate_search_cost"),
        "total": record.get("llm_usage_total"),
        "source": "rq3_optimization_record.json",
        "granularity": "phase aggregate",
    }
