from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from .errors import InfraError
from .isolation import Workspace, artifact_paths
from .util import (ensure_dir, read_json, sha256_file, sha256_value, utc_now,
                   write_json)

CHECKPOINT_VERSION = "scope-exp-checkpoint-v1"
KERNEL_FILE_CANDIDATES = ("cuda_kernel.cuh", "cpu_kernel.c", "main.cpp", "main.c",
                          "kernel.h", "verified_ir.json", "candidate.json")
MAX_HARVEST_BYTES = 64 * 1024 * 1024


def read_artifact(paths: dict[str, Path], name: str) -> dict[str, Any] | None:
    candidate = {
        "top_results": paths["check"] / "top_3_terminal_results.json",
        "performance_unlock": paths["check"] / "performance_unlock_summary.json",
        "final_tuning": paths["check"] / "final_top1_shape_tuning.json",
        "evolution": paths["check"] / "evolution_summary.json",
        "final_code": paths["code"] / "final_code.json",
        "rq2": paths["check"] / "rq2_ablation_record.json",
        "rq3": paths["check"] / "rq3_optimization_record.json",
        "verified_ir": paths["ir_patch"] / "optir.verified.json",
        "shape_specialization": paths["check"] / "shape_specialization.json",
    }.get(name)
    if candidate is None or not candidate.exists():
        return None
    try:
        return read_json(candidate)
    except (ValueError, OSError):
        return None


def collect_artifacts(workspace: Workspace) -> dict[str, Any]:
    paths = artifact_paths(workspace)
    artifacts = {name: read_artifact(paths, name) for name in
                 ("top_results", "performance_unlock", "final_tuning", "evolution",
                  "final_code", "rq2", "rq3", "verified_ir", "shape_specialization")}
    artifacts["paths"] = {key: str(value) for key, value in paths.items()}
    artifacts["present"] = {name: bool(value) for name, value in artifacts.items()
                            if name not in ("paths", "present")}
    return artifacts


def _resolve_within(workspace: Workspace, value: Any) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = workspace.package_dir / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.exists():
        return None
    try:
        resolved.relative_to(workspace.package_dir.resolve())
    except ValueError:
        return None
    return resolved


def _kernel_identity(directory: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name in KERNEL_FILE_CANDIDATES:
        path = directory / name
        if path.is_file():
            files[name] = sha256_file(path)
            sizes[name] = path.stat().st_size
    entry = None
    for name in ("cuda_kernel.cuh", "cpu_kernel.c"):
        if name in files:
            entry = name
            break
    return {"dir": str(directory), "files": files, "sizes": sizes, "entry": entry,
            "digest": sha256_value(files)}


def p1_selection(artifacts: dict[str, Any]) -> dict[str, Any] | None:
    top = artifacts.get("top_results") or {}
    results = top.get("results") or []
    if not results:
        return None
    best = sorted(results, key=lambda item: item.get("rank") or 999)[0]
    return {
        "phase": "p1",
        "code_dir": best.get("code_dir"),
        "strategy_id": best.get("chain_id"),
        "gflops": best.get("gflops"),
        "latency_ms": best.get("latency_ms"),
        "source_phase": best.get("source_phase"),
        "rank": best.get("rank"),
        "correctness_status": best.get("correctness_status"),
        "compile_status": best.get("compile_status"),
        "top_k": [{"rank": item.get("rank"), "strategy_id": item.get("chain_id"),
                   "gflops": item.get("gflops"), "code_dir": item.get("code_dir")}
                  for item in results],
        "selection_rule": "the frozen method's rank-1 terminal candidate; the list "
                          "is captured so the frontier is not lost",
    }


def p2_selection(artifacts: dict[str, Any]) -> dict[str, Any] | None:
    summary = artifacts.get("performance_unlock")
    if not summary:
        return None
    if summary.get("status") in ("skipped", "not_triggered"):
        return None
    return {
        "phase": "p2",
        "code_dir": summary.get("best_code_dir"),
        "strategy_id": summary.get("best_strategy_id"),
        "gflops": summary.get("best_gflops"),
        "profile": summary.get("profile"),
        "frontier_state_count": summary.get("frontier_state_count"),
        "stage_order": summary.get("stage_order"),
        "selection_rule": "feedback-optimisation best after rollback, as reported by "
                          "the frozen application",
    }


def p3_selection(artifacts: dict[str, Any], shape: int) -> dict[str, Any] | None:
    report = artifacts.get("final_tuning")
    if not report:
        return None
    if report.get("status") in ("skipped", "skipped_no_verified_top1"):
        return None
    for item in report.get("shapes") or []:
        declared = item.get("shape") or {}
        if int(declared.get("M") or 0) != int(shape):
            continue
        return {
            "phase": "p3",
            "code_dir": item.get("final_code_dir"),
            "tuning_dir": item.get("tuning_dir"),
            "status": item.get("status"),
            "gflops": item.get("gflops"),
            "latency_ms": item.get("latency_ms"),
            "seed_gflops": item.get("seed_gflops"),
            "stop_reason": item.get("stop_reason"),
            "search_cost": item.get("search_cost"),
            "padding_mode": item.get("padding_mode"),
            "tile": item.get("tile"),
            "relative_to_cublas": item.get("relative_to_cublas"),
            "selection_rule": "the parameter search's own winner for this shape; "
                              "the search limits are recorded with it",
        }
    return None


def search_accounting(artifacts: dict[str, Any], *, config: dict[str, Any] | None = None,
                      shape: int = 0) -> dict[str, Any]:
    report = artifacts.get("final_tuning") or {}
    item = None
    for candidate in report.get("shapes") or []:
        if int((candidate.get("shape") or {}).get("M") or 0) == int(shape):
            item = candidate
            break
    cost = (item or {}).get("search_cost") or {}
    tuning = (config or {}).get("final_top1_tuning") or {}
    max_candidates = tuning.get("max_candidates_per_shape")
    rounds = tuning.get("rounds")
    legal = cost.get("legal_candidates")
    space_total = legal if isinstance(legal, int) else (
        int(max_candidates) * int(rounds) if isinstance(max_candidates, int)
        and isinstance(rounds, int) else None)
    omitted = cost.get("budget_omitted_candidates")
    exhaustive = (isinstance(omitted, int) and omitted == 0 and isinstance(legal, int)
                  and legal >= 0)
    return {
        "shape": shape,
        "space_total": space_total,
        "space_total_source": ("legal_candidates" if isinstance(legal, int)
                               else "max_candidates_per_shape x rounds"),
        "proposed": int(max_candidates) if isinstance(max_candidates, int) else None,
        "attempted": cost.get("tested_candidates"),
        "compiled": cost.get("compile_attempts"),
        "runtime_safe": None,
        "numeric_pass": cost.get("accepted_candidates"),
        "measured": cost.get("benchmark_process_runs"),
        "budget_omitted": omitted,
        "rounds_executed": cost.get("rounds_executed"),
        "rounds_configured": rounds,
        "stop_reason": (item or {}).get("stop_reason") or report.get("stop_reason"),
        "exhaustive": exhaustive,
        "claim": ("exhaustive" if exhaustive else
                  "budget-limited search; not exhaustive"),
        "notes": [
            "runtime_safe is not recorded separately by the frozen tuner",
            "a prefix-truncated candidate list is never reported as exhaustive",
        ],
    }


def capture(paths_root: Path, *, phase: str, selection: dict[str, Any],
            workspace: Workspace, protocol: dict[str, Any],
            config_hash: str | None, app_argv: Sequence[str],
            attempt_id: str, parent: dict[str, Any] | None = None,
            artifacts: dict[str, Any] | None = None,
            config: dict[str, Any] | None = None,
            shape: int = 0,
            harvest_tuning: bool = True) -> dict[str, Any]:
    if phase not in ("p1", "p2", "p3"):
        raise ValueError(f"unknown checkpoint phase: {phase}")
    kernel_dir = _resolve_within(workspace, selection.get("code_dir"))
    if kernel_dir is None:
        return {"phase": phase, "status": "missing_kernel_dir",
                "declared_dir": selection.get("code_dir"),
                "selection": selection,
                "reason": "the recorded code directory does not exist inside the "
                          "isolated workspace"}

    identity = _kernel_identity(kernel_dir)
    if not identity["files"]:
        return {"phase": phase, "status": "empty_kernel_dir",
                "declared_dir": str(kernel_dir), "selection": selection}

    verified_ir = kernel_dir / "verified_ir.json"
    ir_hash = sha256_file(verified_ir) if verified_ir.is_file() else None
    artifact_hash = sha256_value({
        "phase": phase,
        "kernel": identity["files"],
        "ir": ir_hash,
        "strategy_id": selection.get("strategy_id"),
        "source_phase": selection.get("source_phase"),
        "task_id": workspace.task_id,
        "parent": (parent or {}).get("artifact_hash"),
    })
    destination = Path(paths_root) / phase / artifact_hash
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        return read_json(manifest_path)

    ensure_dir(destination / "kernel")
    for name in identity["files"]:
        source_file = kernel_dir / name
        target = destination / "kernel" / name
        target.write_bytes(source_file.read_bytes())

    harvested: dict[str, Any] = {}
    if harvest_tuning and selection.get("tuning_dir"):
        tuning_dir = _resolve_within(workspace, selection.get("tuning_dir"))
        if tuning_dir is not None:
            harvested = _harvest_dir(tuning_dir, destination / "tuning",
                                     limit_bytes=MAX_HARVEST_BYTES)

    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "phase": phase,
        "task_id": workspace.task_id,
        "attempt_id": attempt_id,
        "created_at": utc_now(),
        "artifact_hash": artifact_hash,
        "status": "captured",
        "kernel": {"entry": identity["entry"], "files": identity["files"],
                   "sizes": identity["sizes"], "digest": identity["digest"],
                   "source_dir": str(kernel_dir)},
        "ir": {"path": str(verified_ir) if ir_hash else None, "sha256": ir_hash},
        "selection": selection,
        "provenance": {
            "app_argv": list(app_argv),
            "config_hash": config_hash,
            "protocol_hash": protocol.get("protocol_hash"),
            "harness_version": protocol.get("harness_version"),
            "workspace": str(workspace.root),
            "source_digest": workspace.source_digest,
        },
        "parent": parent,
        "harvested": harvested,
        "search_accounting": search_accounting(artifacts or {}, config=config,
                                               shape=shape),
        "immutability": "content addressed; an existing directory is never rewritten",
    }
    write_json(manifest_path, manifest)
    return manifest


def _harvest_dir(source: Path, destination: Path, *, limit_bytes: int) -> dict[str, Any]:
    ensure_dir(destination)
    total = 0
    files: dict[str, str] = {}
    skipped: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        if total + size > limit_bytes:
            skipped.append(path.relative_to(source).as_posix())
            continue
        target = destination / path.relative_to(source)
        ensure_dir(target.parent)
        target.write_bytes(path.read_bytes())
        files[path.relative_to(source).as_posix()] = sha256_file(target)
        total += size
    return {"files": files, "bytes": total, "skipped": skipped[:32],
            "skipped_count": len(skipped), "source": str(source)}


def select_delivered(artifacts: dict[str, Any], shape: int) -> dict[str, Any]:
    p3 = p3_selection(artifacts, shape)
    p2 = p2_selection(artifacts)
    p1 = p1_selection(artifacts)
    order: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    if p3 and p3.get("status") == "completed" and p3.get("code_dir"):
        chosen = p3
        rule = "P3 completed for this shape"
    elif p2 and p2.get("code_dir"):
        chosen = p2
        rule = "P2 rollback-aware best (P3 did not complete)"
    elif p1 and p1.get("code_dir"):
        chosen = p1
        rule = "P1 rank-1 terminal candidate (P2/P3 produced nothing better)"
    else:
        rule = "no phase produced a kernel"
    for candidate, name in ((p3, "p3"), (p2, "p2"), (p1, "p1")):
        order.append({"phase": name,
                      "present": bool(candidate),
                      "code_dir": (candidate or {}).get("code_dir"),
                      "gflops": (candidate or {}).get("gflops")})
    return {
        "delivered_phase": (chosen or {}).get("phase"),
        "delivered": chosen is not None,
        "rule": rule,
        "selection": chosen,
        "precedence": ["p3_completed_shape", "p2_rollback_best", "p1_rank1"],
        "candidate_order": order,
        "frozen_at": utc_now(),
        "no_reselection_after_validation": True,
    }


def find_kernel_candidates(workspace: Workspace) -> list[Path]:
    roots = [workspace.package_dir / "gemm_code", workspace.package_dir / "results"]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            if (path / "cuda_kernel.cuh").is_file() or (path / "cpu_kernel.c").is_file():
                found.append(path)
    return sorted(set(found))


def require_kernel_dir(workspace: Workspace, selection: dict[str, Any]) -> Path:
    directory = _resolve_within(workspace, selection.get("code_dir"))
    if directory is None:
        raise InfraError(f"selected kernel directory is not usable: "
                         f"{selection.get('code_dir')}", stage="checkpoint")
    return directory
