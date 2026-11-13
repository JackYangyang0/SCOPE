from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from . import HARNESS_VERSION
from .paths import CampaignPaths
from .redact import find_credentials, live_credentials
from .util import (ensure_dir, read_json, read_jsonl, sha256_file, sha256_value,
                   utc_now, write_csv, write_json)

AUDIT_VERSION = "scope-exp-audit-v1"
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".yaml", ".yml"}


def _planned_tasks(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not protocol:
        return []
    return list((protocol.get("tasks") or {}).get("task_records") or [])


def _status_of(paths: CampaignPaths, task_id: str) -> dict[str, Any] | None:
    path = paths.task_status(task_id)
    return read_json(path) if path.exists() else None


def check_completeness(paths: CampaignPaths, protocol: dict[str, Any] | None) -> dict[str, Any]:
    planned = _planned_tasks(protocol)
    statuses: dict[str, int] = {}
    missing: list[str] = []
    unexpected: list[str] = []
    known = {record["task_id"] for record in planned}
    for record in planned:
        status = _status_of(paths, record["task_id"])
        if status is None:
            missing.append(record["task_id"])
            statuses["planned"] = statuses.get("planned", 0) + 1
            continue
        name = status.get("status", "unknown")
        statuses[name] = statuses.get(name, 0) + 1
    for status_path in sorted(Path(paths.tasks).glob("*/status.json")):
        task_id = status_path.parent.name
        if planned and task_id not in known:
            unexpected.append(task_id)
    problems: list[str] = []
    if unexpected:
        problems.append(f"task directories not in the frozen manifest: {unexpected[:8]}")
    if planned and missing:
        problems.append(f"{len(missing)} planned tasks have no status file")
    return {
        "planned": len(planned),
        "independent": sum(1 for record in planned
                           if record.get("kind") == "independent"),
        "branches": sum(1 for record in planned if record.get("kind") == "branch"),
        "aliases": sum(1 for record in planned if record.get("kind") == "alias"),
        "status_counts": dict(sorted(statuses.items())),
        "missing_status_files": missing[:32],
        "unexpected_task_dirs": unexpected[:32],
        "resolved": statuses.get("resolved", 0),
        "unresolved": len(planned) - statuses.get("resolved", 0),
        "problems": problems,
        "ok": not problems,
    }


def check_identity(paths: CampaignPaths) -> dict[str, Any]:
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    for status_path in sorted(Path(paths.tasks).glob("*/status.json")):
        task_id = status_path.parent.name
        status = read_json(status_path)
        if status.get("status") != "resolved":
            continue
        selected_path = paths.task_selected(task_id)
        selected = read_json(selected_path) if selected_path.exists() else None
        validation_path = paths.task_validation_json(task_id)
        validation = read_json(validation_path) if validation_path.exists() else None
        timing_path = paths.task_timing_json(task_id)
        timing = read_json(timing_path) if timing_path.exists() else None
        problems_here: list[str] = []
        delivered_flag = bool(((status.get("outcome") or {}).get("delivered")))
        if selected is None:
            problems_here.append("no selected.json")
        if validation is None:
            problems_here.append("no validation.json")
        if timing is None and delivered_flag:
            problems_here.append("no timing.json for a delivered task")
        validation_binary = ((validation or {}).get("summary") or {}).get("binary_sha256")
        timing_binary = ((timing or {}).get("spec") or {}).get("binary_sha256")
        if validation_binary and timing_binary and validation_binary != timing_binary:
            problems_here.append("validation and timing used different executables")
        delivered = ((selected or {}).get("delivered") or {})
        phase = delivered.get("delivered_phase")
        checkpoint_hash = None
        if phase:
            manifests = sorted((paths.task_dir(task_id) / "checkpoints" / phase)
                               .glob("*/manifest.json"))
            if manifests:
                checkpoint_hash = read_json(manifests[-1]).get("artifact_hash")
            else:
                problems_here.append(f"no checkpoint manifest for delivered phase {phase}")
        if status.get("delivered_phase") and phase and status["delivered_phase"] != phase:
            problems_here.append("status and selected.json disagree on the delivered phase")
        if not status.get("resolved_config_hash"):
            problems_here.append("no resolved config hash")
        problems.extend(f"{task_id}: {item}" for item in problems_here)
        rows.append({"task_id": task_id, "delivered_phase": phase,
                     "checkpoint_hash": checkpoint_hash,
                     "validation_binary": validation_binary,
                     "timing_binary": timing_binary,
                     "same_binary": bool(validation_binary and
                                         validation_binary == timing_binary),
                     "config_hash": status.get("resolved_config_hash"),
                     "problems": problems_here})
    return {"ok": not problems, "problems": problems, "tasks": rows,
            "note": "validation and timing reuse one executable, so kernel object "
                    "identity is proven by binary identity (§10.2)"}


def check_no_reselection(paths: CampaignPaths) -> dict[str, Any]:
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    for selected_path in sorted(Path(paths.tasks).glob("*/selected.json")):
        task_id = selected_path.parent.name
        selected = read_json(selected_path)
        frozen_at = selected.get("frozen_at")
        validation_path = paths.task_validation_json(task_id)
        validation = read_json(validation_path) if validation_path.exists() else None
        started = (validation or {}).get("started_at")
        ordered = True
        if frozen_at and started:
            ordered = str(frozen_at) <= str(started)
        events: list[dict[str, Any]] = []
        for event_file in sorted(Path(paths.task_dir(task_id) / "attempts")
                                 .glob("*/events.jsonl")):
            events.extend(read_jsonl(event_file))
        top1_events = [event for event in events if event.get("event") == "top1_frozen"]
        selection_hashes = {event.get("selection_sha256") for event in top1_events
                            if event.get("selection_sha256")}
        if len(selection_hashes) > 1:
            problems.append(f"{task_id}: the frozen Top-1 changed between attempts "
                            f"({len(selection_hashes)} distinct selections)")
        if not ordered:
            problems.append(f"{task_id}: validation started before the Top-1 was frozen")
        rows.append({"task_id": task_id, "frozen_at": frozen_at,
                     "validation_started": started, "ordered": ordered,
                     "top1_frozen_events": len(top1_events),
                     "selected_sha256": sha256_value(selected)})
    return {"ok": not problems, "problems": problems, "tasks": rows}


def check_cost(paths: CampaignPaths) -> dict[str, Any]:
    problems: list[str] = []
    total_rows = 0
    unknown_cost_tasks: list[str] = []
    tasks_without_ledger: list[str] = []
    for status_path in sorted(Path(paths.tasks).glob("*/status.json")):
        task_id = status_path.parent.name
        status = read_json(status_path)
        if status.get("status") not in ("resolved", "no_candidate", "generation_failed"):
            continue
        rows: list[dict[str, Any]] = []
        for ledger_file in sorted(status_path.parent.glob("attempts/*/api_ledger.jsonl")):
            rows.extend(read_jsonl(ledger_file))
        total_rows += len(rows)
        reported = (status.get("cost") or {}).get("requests")
        if reported is None:
            if rows:
                problems.append(f"{task_id}: ledger rows exist but no cost summary")
            continue
        if reported != len(rows):
            problems.append(f"{task_id}: cost summary says {reported} requests, "
                            f"ledger has {len(rows)}")
        if rows and not (status.get("cost") or {}).get("cost_is_complete"):
            unknown_cost_tasks.append(task_id)
        if not rows and status.get("status") == "resolved":
            tasks_without_ledger.append(task_id)
    return {
        "ok": not problems,
        "problems": problems,
        "ledger_rows": total_rows,
        "tasks_with_unknown_cost": unknown_cost_tasks[:32],
        "resolved_tasks_without_ledger": tasks_without_ledger[:32],
        "note": "a resolved task with no ledger means requests were not observable; "
                "that is reported, not filled with zero",
    }


def check_inputs(paths: CampaignPaths, protocol: dict[str, Any] | None, *,
                 deep: bool = False) -> dict[str, Any]:
    from . import datasets
    if not protocol:
        return {"ok": False, "problems": ["no protocol"], "splits": {}}
    problems: list[str] = []
    splits: dict[str, Any] = {}
    recorded = ((protocol.get("data") or {}).get("manifest_hashes") or {})
    for split, frozen_hash in recorded.items():
        path = paths.dataset_dir(split) / "manifest.json"
        if not path.exists():
            problems.append(f"{split}: manifest missing")
            continue
        manifest = read_json(path)
        if manifest.get("manifest_hash") != frozen_hash:
            problems.append(f"{split}: manifest hash changed")
        entry = {"manifest": str(path), "frozen_hash": frozen_hash,
                 "current_hash": manifest.get("manifest_hash"),
                 "entries": manifest.get("entry_count")}
        if deep:
            result = datasets.verify_dataset(paths, split, jobs=1)
            entry["deep_verification"] = result
            if not result["ok"]:
                problems.append(f"{split}: {result['problems']}")
        splits[split] = entry
    return {"ok": not problems, "problems": problems, "splits": splits,
            "deep": deep}


def check_branches(paths: CampaignPaths, protocol: dict[str, Any] | None) -> dict[str, Any]:
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    planned = [record for record in _planned_tasks(protocol)
               if record.get("block") == "E1" and record.get("kind") in ("branch", "alias")]
    for record in planned:
        kind = ("without_feedback" if record["kind"] == "branch"
                else "without_parameter_search")
        path = paths.branch_dir(record["parent_task_id"], kind) / "branch.json"
        if not path.exists():
            rows.append({"branch_task_id": record["task_id"], "kind": kind,
                         "status": "not_executed",
                         "parent_task_id": record["parent_task_id"]})
            continue
        payload = read_json(path)
        status = payload.get("status")
        if status not in ("completed", "parent_failed", "branch_failed", "started"):
            problems.append(f"{record['task_id']}: unknown branch status {status}")
        parent_status = _status_of(paths, record["parent_task_id"])
        if parent_status is None:
            problems.append(f"{record['task_id']}: parent has no status")
        elif status == "completed" and parent_status.get("status") in (
                "no_candidate", "generation_failed"):
            problems.append(f"{record['task_id']}: branch completed although the "
                            "parent failed")
        rows.append({"branch_task_id": record["task_id"], "kind": kind,
                     "status": status, "parent_task_id": record["parent_task_id"],
                     "parent_status": (parent_status or {}).get("status")})
    completed = sum(1 for row in rows if row["status"] == "completed")
    no_feedback = sum(1 for row in rows if row["kind"] == "without_feedback")
    if no_feedback > 40:
        problems.append(f"{no_feedback} no-feedback branches planned; the design "
                        "allows at most 40")
    return {"ok": not problems, "problems": problems, "branches": rows,
            "planned": len(rows), "completed": completed,
            "no_feedback_count": no_feedback,
            "independent_n_from_branches": 0}


def check_redaction(paths: CampaignPaths, *, limit: int = 4000,
                    protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    live = live_credentials()
    try:
        from . import config as config_module
        base_path = ((protocol or {}).get("source") or {}).get("base_config_path")
        roles = list(((protocol or {}).get("models") or {}))
        if base_path and roles:
            base = config_module.load_yaml(Path(base_path))
            live = list(live) + config_module.credential_values_for_scan(base, roles)
    except Exception:
        pass
    hits: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(Path(paths.root).rglob("*")):
        if scanned >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
        except OSError:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = find_credentials(text, live)
        if found:
            hits.append({"path": str(path.relative_to(paths.root)), "findings": found})
    return {"ok": not hits, "scanned_files": scanned, "hits": hits[:32],
            "note": "campaign archives never contain conf.yaml or an API key"}


def check_claims(paths: CampaignPaths) -> dict[str, Any]:
    summary_path = paths.analysis / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    tables: dict[str, Any] = {}
    for name in ("rq2_components", "rq2_components_pooled", "models", "models_pooled",
                 "stage_pairs", "correctness", "cost", "completeness"):
        csv_path = paths.analysis / f"{name}.csv"
        rows = 0
        if csv_path.exists():
            rows = max(0, len(csv_path.read_text(encoding="utf-8").splitlines()) - 1)
        tables[name] = {
            "csv": str(csv_path) if csv_path.exists() else None,
            "rows": rows,
            "regenerable": csv_path.exists(),
            "source": "analysis/summary.json <- tasks/*/status.json + validation.json "
                      "+ timing.json + attempts/*/api_ledger.jsonl",
            "present_in_summary": bool(summary and name in (summary.get("tables") or {})),
        }
    problems: list[str] = []
    if summary is None:
        problems.append("analysis/summary.json is missing; run `scope-exp summarize`")
    else:
        for name, record in tables.items():
            if record["present_in_summary"] and not record["regenerable"]:
                rows_in_summary = len(((summary.get("tables") or {}).get(name) or []))
                if rows_in_summary:
                    problems.append(f"{name}: summary has {rows_in_summary} row(s) but "
                                    "no CSV was written")
    claims = {
        "created_at": utc_now(),
        "tables": tables,
        "claims": [
            {"claim": "C1",
             "question": "how many tasks deliver a numerically valid, competitive kernel",
             "tables": ["rq2_components", "models", "correctness"],
             "sources": "independent revalidation (64 main + 128 coefficient checks) "
                        "and the frozen timing protocol"},
            {"claim": "C2",
             "question": "do structured state, local checks, stage ordering, feedback "
                         "and parameter search contribute",
             "tables": ["rq2_components", "stage_pairs"],
             "sources": "same-version ablation and shared-parent stage pairs"},
        ],
        "limitations": [
            "the delivered count is complete delivery (validation + applicable safety); "
            "a missing sanitizer run is not a pass",
            "stage pairs add no independent N",
            "extensions (D1/E3/cross-platform) are reported separately and never "
            "merged into the core 280",
        ],
    }
    return {"ok": not problems, "problems": problems, "claims": claims}


def build_index(paths: CampaignPaths, protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _planned_tasks(protocol):
        task_id = record["task_id"]
        status = _status_of(paths, task_id) or {}
        selected_path = paths.task_selected(task_id)
        rows.append({
            "task_id": task_id,
            "block": record.get("block"),
            "kind": record.get("kind"),
            "shape": record.get("shape_M"),
            "variant": record.get("variant"),
            "model_role": record.get("model_role"),
            "status": status.get("status", "planned"),
            "status_path": str(paths.task_status(task_id)),
            "status_sha256": (sha256_file(paths.task_status(task_id))
                              if paths.task_status(task_id).exists() else None),
            "selected_path": str(selected_path) if selected_path.exists() else None,
            "selected_sha256": sha256_file(selected_path) if selected_path.exists() else None,
            "validation_csv": str(paths.task_validation_csv(task_id))
            if paths.task_validation_csv(task_id).exists() else None,
            "timing_csv": str(paths.task_timing_csv(task_id))
            if paths.task_timing_csv(task_id).exists() else None,
            "config_hash": status.get("resolved_config_hash"),
            "delivered_phase": status.get("delivered_phase"),
            "provenance": status.get("provenance"),
        })
    return rows


def run(paths: CampaignPaths, *, protocol: dict[str, Any] | None = None,
        deep_inputs: bool = False) -> dict[str, Any]:
    from . import ablation_audit, controls
    completeness = check_completeness(paths, protocol)
    identity = check_identity(paths)
    reselection = check_no_reselection(paths)
    cost = check_cost(paths)
    inputs = check_inputs(paths, protocol, deep=deep_inputs)
    branches = check_branches(paths, protocol)
    redaction = check_redaction(paths, protocol=protocol)
    claims = check_claims(paths)
    controls_state = controls.gate_status(paths)
    ablation_state = ablation_audit.audit_status(paths)
    protocol_state = {
        "present": bool(protocol),
        "frozen": bool(protocol and protocol.get("status") == "frozen"),
        "protocol_hash": (protocol or {}).get("protocol_hash"),
        "backend": (protocol or {}).get("backend"),
        "denominators": ((protocol or {}).get("tasks") or {}).get("denominators"),
    }
    abandoned_path = paths.audit / "abandoned.jsonl"
    abandoned = read_jsonl(abandoned_path) if abandoned_path.exists() else []
    sections = {"completeness": completeness, "identity": identity,
                "abandoned_retries": {
                    "count": len(abandoned),
                    "tasks": [row.get("task_id") for row in abandoned][:32],
                    "ok": True,
                    "problems": [],
                    "note": "a harness repair archives the failed attempts and re-runs "
                            "the same logical slot; it adds no independent N and is "
                            "listed here so the retry is auditable",
                },
                "no_reselection": reselection, "cost": cost, "inputs": inputs,
                "branches": branches, "redaction": redaction}
    problems: list[str] = []
    for name, section in sections.items():
        problems.extend(f"{name}: {item}" for item in section.get("problems") or [])
    if not controls_state.get("ok"):
        problems.append("controls: the E0 gate is missing or red")
    if ablation_state.get("present") and not ablation_state.get("ok"):
        problems.append("ablation: the diff/leakage audit failed")
    if not protocol_state["frozen"]:
        problems.append("protocol: no frozen protocol loaded")
    problems.extend(f"claims: {item}" for item in claims.get("problems") or [])

    index_rows = build_index(paths, protocol)
    ensure_dir(paths.audit)
    write_csv(paths.audit / "index.csv", index_rows)
    write_json(paths.audit / "claims.json", claims["claims"])
    report = {
        "audit_version": AUDIT_VERSION,
        "harness_version": HARNESS_VERSION,
        "created_at": utc_now(),
        "campaign_id": paths.campaign_id,
        "protocol": protocol_state,
        "controls_gate": controls_state,
        "ablation_audit": ablation_state,
        "sections": sections,
        "index_rows": len(index_rows),
        "index_path": str(paths.audit / "index.csv"),
        "claims_path": str(paths.audit / "claims.json"),
        "problems": problems,
        "ok": not problems,
        "completeness_note": "execution completeness and scientific support are "
                             "reported separately: a complete archive can still "
                             "support no claim",
    }
    write_json(paths.audit / "report.json", report)
    return report


def audit_status(paths: CampaignPaths) -> dict[str, Any]:
    path = paths.audit / "report.json"
    if not path.exists():
        return {"present": False, "path": str(path)}
    record = read_json(path)
    return {"present": True, "ok": record.get("ok"),
            "problems": record.get("problems"),
            "created_at": record.get("created_at"), "path": str(path)}
