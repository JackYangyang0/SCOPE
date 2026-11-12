
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

from . import HARNESS_VERSION
from .errors import ConfigError, ProtocolError
from .paths import LAYOUT_VERSION, CampaignPaths
from .util import read_json, sha256_value, utc_now, write_json

PROTOCOL_VERSION = "scope-exp-protocol-v1"

REQUIRED_TOP_LEVEL = (
    "protocol_version", "campaign_id", "status", "harness_version",
    "source", "environment", "backend", "shapes", "repeats", "blocks",
    "tasks", "models", "data", "validation", "timing", "budget",
    "selection_rules", "statistics", "controls_gate", "sanitizer",
)

REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "source": ("repo_root", "git_head", "git_dirty", "source_digest",
               "file_count", "method_files"),
    "environment": ("python", "platform", "numpy_version", "toolchain"),
    "tasks": ("counts", "task_ids", "manifest_sha256", "order", "blocks"),
    "data": ("generator_version", "splits", "checks_per_kernel_per_shape",
             "coefficient_sets", "manifest_hashes"),
    "validation": ("policies", "atol", "rtol", "relative_l2", "strict_abs",
                   "reference", "timeout_seconds", "repeat_checks",
                   "hypotheses_max_shape"),
    "timing": ("warmup_processes", "measure_processes", "launch_warmups",
               "launch_iters", "primary_statistic", "throughput_formula",
               "trimming_rule"),
    "budget": ("per_task_wall_seconds", "per_phase_wall_seconds",
               "api_timeout_seconds", "max_retries", "search", "stop_rules",
               "budget_profile", "pilot_wall_seconds"),
    "selection_rules": ("top1", "no_reselection_after_validation",
                        "without_feedback_start", "budget_exhaustion"),
    "statistics": ("alpha", "bootstrap_resamples", "bootstrap_seed",
                   "wilson_intervals", "paired_by_parent", "unit_of_analysis"),
    "controls_gate": ("path", "sha256", "backend", "ok"),
    "sanitizer": ("timeout_seconds", "tools"),
}

STATUSES = ("draft", "frozen", "superseded")


def blank_protocol(campaign_id: str, *, backend: str = "cuda",
                   shapes: Sequence[int] = (512, 1024, 2048, 4096),
                   repeats: int = 10, blocks: Sequence[str] = ("E1", "E2")) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "campaign_id": campaign_id,
        "status": "draft",
        "harness_version": HARNESS_VERSION,
        "layout_version": LAYOUT_VERSION,
        "created_at": utc_now(),
        "backend": backend,
        "shapes": [int(shape) for shape in shapes],
        "repeats": int(repeats),
        "blocks": list(blocks),
        "source": {},
        "environment": {},
        "tasks": {},
        "models": {},
        "data": {},
        "validation": {},
        "timing": {},
        "budget": {},
        "selection_rules": {},
        "statistics": {},
        "controls_gate": {},
        "sanitizer": {},
        "extensions": {
            "dependency_revision": {"planned": False, "block": "D1",
                                    "note": "needs scope-exp-dependency-v2; not part "
                                            "of the core 280"},
            "budget_efficiency": {"planned": False, "block": "E3"},
            "cross_platform": {"planned": False, "note": "H800 / 14700KF are fixed-artifact "
                                                          "audits, not generation"},
        },
    }


def validate(protocol: dict[str, Any], *, require_frozen: bool = True,
             paths: CampaignPaths | None = None,
             allow_missing_credentials: bool = False) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any, why: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail, "why": why})

    problems: list[str] = []

    for key in REQUIRED_TOP_LEVEL:
        present = key in protocol
        record(f"top_level.{key}", present, bool(present), "required protocol field")
        if not present:
            problems.append(f"missing top-level field {key}")
    for section, fields in REQUIRED_SECTIONS.items():
        values = protocol.get(section) or {}
        for field in fields:
            present = field in values
            record(f"{section}.{field}", present, bool(present),
                   "required protocol sub-field")
            if not present:
                problems.append(f"missing {section}.{field}")

    status = protocol.get("status")
    record("status", status in STATUSES, status, "status must be one of the known states")
    if require_frozen and status != "frozen":
        problems.append(f"protocol status is {status!r}; run requires 'frozen'")

    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        problems.append(f"protocol_version {protocol.get('protocol_version')} != "
                        f"{PROTOCOL_VERSION}")

    budget = protocol.get("budget") or {}
    for field in ("per_task_wall_seconds", "api_timeout_seconds", "pilot_wall_seconds"):
        value = budget.get(field)
        ok = isinstance(value, (int, float)) and value > 0
        record(f"budget.{field}", ok, value, "a formal queue may not start without a limit")
        if not ok:
            problems.append(f"budget.{field} must be a positive number, got {value!r}")
    record("budget.budget_profile", budget.get("budget_profile") == "default",
           budget.get("budget_profile"),
           "compact mode rewrites search width and target shapes")
    if budget.get("budget_profile") != "default":
        problems.append("budget_profile must be 'default' for a formal queue")
    search = budget.get("search") or {}
    for field in ("max_candidates_per_shape", "rounds", "max_frontier_states_per_stage",
                  "max_total_compilations", "tiling_resource_pool_size"):
        ok = isinstance(search.get(field), int) and search.get(field, 0) > 0
        record(f"budget.search.{field}", ok, search.get(field),
               "search limits are part of the comparison, not an afterthought")
        if not ok:
            problems.append(f"budget.search.{field} must be a positive integer")
    if not budget.get("stop_rules"):
        problems.append("budget.stop_rules must be non-empty")

    source = protocol.get("source") or {}
    digest = source.get("source_digest")
    ok = isinstance(digest, str) and len(digest) == 64
    record("source.source_digest", ok, digest,
           "the frozen source snapshot must carry a content digest")
    if not ok:
        problems.append("source.source_digest is missing or not a sha256 digest")
    ok = isinstance(source.get("method_files"), dict) and bool(source["method_files"])
    record("source.method_files", ok, len(source.get("method_files") or {}),
           "the protocol must freeze the method files it will run")
    if not ok:
        problems.append("source.method_files is empty")

    data = protocol.get("data") or {}
    if not data.get("manifest_hashes"):
        problems.append("data.manifest_hashes is empty; the frozen inputs are unknown")

    models = protocol.get("models") or {}
    if not models:
        problems.append("models is empty; resolve at least the E2 roles")
    for role, entry in models.items():
        resolved = bool(entry.get("model")) and "REPLACE" not in str(entry.get("model"))
        record(f"models.{role}.resolved", resolved, entry.get("model"),
               "an unresolved model id must stop the run (§13 REQ-01)")
        if not resolved:
            problems.append(f"model {role} is unresolved")
        credential = bool(entry.get("api_key_env_present")
                          or entry.get("api_key_present_inline"))
        source = entry.get("api_key_source") or ("environment" if entry.get(
            "api_key_env_present") else "config_file" if entry.get(
            "api_key_present_inline") else "none")
        record(f"models.{role}.credential", credential or allow_missing_credentials,
               source, "the key is injected into the child environment at run time")
        if not credential and not allow_missing_credentials:
            problems.append(f"model {role}: no credential (environment variable "
                            f"{entry.get('api_key_env')!r} is unset and api_key is "
                            "empty)")

    data = protocol.get("data") or {}
    checks_per_shape = data.get("checks_per_kernel_per_shape") or {}
    record("data.checks.main", checks_per_shape.get("main") == 64,
           checks_per_shape.get("main"), "32 configs x 2 C0 variants")
    record("data.checks.coefficients", checks_per_shape.get("coefficients") == 128,
           checks_per_shape.get("coefficients"), "2 coefficient sets x 32 x 2")
    if checks_per_shape.get("main") != 64:
        problems.append("data.checks_per_kernel_per_shape.main must be 64")

    validation = protocol.get("validation") or {}
    from . import criteria
    for field, expected in (("atol", criteria.ATOL), ("rtol", criteria.RTOL),
                            ("relative_l2", criteria.RELATIVE_L2_TOLERANCE),
                            ("strict_abs", criteria.STRICT_ABS_TOLERANCE)):
        ok = validation.get(field) == expected
        record(f"validation.{field}", ok, validation.get(field),
               "tolerances are frozen constants")
        if not ok:
            problems.append(f"validation.{field} {validation.get(field)!r} != {expected}")

    timing = protocol.get("timing") or {}
    from . import timing as timing_module
    for field, expected in (("warmup_processes", timing_module.TIMING_WARMUP_PROCESSES),
                            ("measure_processes", timing_module.TIMING_MEASURE_PROCESSES),
                            ("launch_warmups", timing_module.LAUNCH_WARMUPS),
                            ("launch_iters", timing_module.LAUNCH_ITERS)):
        ok = timing.get(field) == expected
        record(f"timing.{field}", ok, timing.get(field), "timing protocol is frozen")
        if not ok:
            problems.append(f"timing.{field} {timing.get(field)!r} != {expected}")

    tasks = protocol.get("tasks") or {}
    task_ids = tasks.get("task_ids") or []
    record("tasks.non_empty", bool(task_ids), len(task_ids), "no tasks, no campaign")
    if not task_ids:
        problems.append("tasks.task_ids is empty")
    if tasks.get("manifest_sha256") and task_ids:
        recomputed = sha256_value(task_ids)
        ok = recomputed == tasks["manifest_sha256"]
        record("tasks.manifest_sha256", ok, recomputed,
               "the task list must not change after freezing")
        if not ok:
            problems.append("tasks.manifest_sha256 does not match the task id list")

    gate = protocol.get("controls_gate") or {}
    ok = bool(gate.get("ok")) and gate.get("backend") == protocol.get("backend")
    record("controls_gate", ok, {"ok": gate.get("ok"),
                                "backend": gate.get("backend")},
           "E0 must be green for this backend before any task runs")
    if not ok:
        problems.append("controls gate is missing, red, or for another backend")

    if paths is not None:
        from . import datasets
        for split, recorded in (data.get("manifest_hashes") or {}).items():
            path = paths.dataset_dir(split) / "manifest.json"
            exists = path.exists()
            record(f"data.manifest_file.{split}", exists, str(path),
                   "the frozen manifest must still be on disk")
            if not exists:
                problems.append(f"dataset manifest for {split} is missing")
                continue
            actual = read_json(path).get("manifest_hash")
            ok = actual == recorded
            record(f"data.manifest_hash.{split}", ok, actual,
                   "the manifest that was frozen must be the manifest used")
            if not ok:
                problems.append(f"dataset manifest hash for {split} changed")

    statistics = protocol.get("statistics") or {}
    ok = isinstance(statistics.get("bootstrap_resamples"), int) and \
        statistics.get("bootstrap_resamples", 0) >= 10000
    record("statistics.bootstrap_resamples", ok, statistics.get("bootstrap_resamples"),
           "task-level bootstrap with at least 10,000 resamples")
    if not ok:
        problems.append("statistics.bootstrap_resamples must be >= 10000")

    if problems:
        raise ProtocolError("; ".join(problems))
    return checks


def freeze_checks(protocol: dict[str, Any], paths: CampaignPaths | None = None) -> list[dict[str, Any]]:
    return validate(protocol, require_frozen=False, paths=paths)


def write_protocol(paths: CampaignPaths, protocol: dict[str, Any],
                   *, filename: str = "protocol.json", validate_first: bool = False,
                   require_frozen: bool = True, allow_missing_credentials: bool = False,
                   ) -> dict[str, Any]:
    payload = copy.deepcopy(protocol)
    payload["protocol_hash"] = sha256_value({k: v for k, v in payload.items()
                                             if k not in ("protocol_hash",
                                                          "validation_checks")})
    if validate_first:
        checks = validate(payload, require_frozen=require_frozen, paths=paths,
                          allow_missing_credentials=allow_missing_credentials)
        payload["validation_checks"] = checks
    write_json(paths.root / filename, payload)
    return payload


def load(paths: CampaignPaths, *, frozen: bool = True,
         allow_missing_credentials: bool = False) -> dict[str, Any]:
    name = "FROZEN_PROTOCOL.json" if frozen else "protocol.json"
    path = paths.root / name
    if not path.exists():
        raise ProtocolError(f"{path} does not exist; run "
                            f"`scope-exp {'freeze' if frozen else 'prepare'}` first")
    protocol = read_json(path)
    validate(protocol, require_frozen=frozen, paths=paths,
             allow_missing_credentials=allow_missing_credentials)
    stored = protocol.get("protocol_hash")
    recomputed = sha256_value({k: v for k, v in protocol.items()
                               if k not in ("protocol_hash", "validation_checks")})
    if stored and stored != recomputed:
        raise ProtocolError(f"{path}: protocol_hash mismatch; the file was edited "
                            "after freezing")
    return protocol


def protocol_status(paths: CampaignPaths) -> dict[str, Any]:
    for name in ("FROZEN_PROTOCOL.json", "protocol.json"):
        path = paths.root / name
        if path.exists():
            record = read_json(path)
            return {"path": str(path), "present": True,
                    "status": record.get("status"),
                    "frozen": name.startswith("FROZEN"),
                    "protocol_hash": record.get("protocol_hash"),
                    "campaign_id": record.get("campaign_id"),
                    "created_at": record.get("created_at")}
    return {"present": False}
