from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import yaml
from . import (build, checkpoints, config as config_module, criteria, isolation,
               reference, sanitizer, timing, verify)
from .errors import InfraError
from .paths import CampaignPaths
from .util import (ensure_dir, read_json, read_jsonl, run_command, sha256_file,
                   sha256_value, utc_now, write_json)

BRANCH_VERSION = "scope-exp-branch-v1"


def branch_kinds() -> tuple[str, str]:
    return ("without_feedback", "without_parameter_search")


def _parent_checkpoint(paths: CampaignPaths, parent_id: str, phase: str) -> dict[str, Any] | None:
    directory = paths.task_dir(parent_id) / "checkpoints" / phase
    if not directory.exists():
        return None
    manifests = sorted(directory.glob("*/manifest.json"))
    if not manifests:
        return None
    record = read_json(manifests[-1])
    return record


def _prepare_branch(paths: CampaignPaths, parent_id: str, kind: str) -> tuple[Path, dict[str, Any]]:
    directory = ensure_dir(paths.branch_dir(parent_id, kind))
    status_path = directory / "branch.json"
    if status_path.exists():
        return directory, read_json(status_path)
    record = {
        "branch_version": BRANCH_VERSION,
        "kind": kind,
        "parent_task_id": parent_id,
        "created_at": utc_now(),
        "status": "started",
        "paired_observation": True,
        "independent_n": 0,
        "shared_prefix_cost_counted_once": True,
    }
    write_json(status_path, record)
    return directory, record


def _kernel_from_checkpoint(paths: CampaignPaths, directory: Path,
                            checkpoint: dict[str, Any]) -> Path:
    source = Path(checkpoint.get("kernel", {}).get("source_dir") or "")
    target = ensure_dir(directory / "kernel")
    for name in (checkpoint.get("kernel", {}).get("files") or {}):
        origin = source / name
        if origin.is_file():
            (target / name).write_bytes(origin.read_bytes())
    if not any(target.iterdir()):
        raise InfraError(f"checkpoint {checkpoint.get('artifact_hash')} has no kernel "
                         "files to copy", stage="branch")
    return target


def _validate_and_time(paths: CampaignPaths, protocol: dict[str, Any], kernel_dir: Path,
                       *, backend: str, shape: int, label: str, out_dir: Path,
                       device: int, evidence_dir: Path) -> dict[str, Any]:
    build_result = build.build_eval_driver(kernel_dir, out_dir / "build", backend=backend)
    if not build_result.ok:
        return {"status": "build_failed", "build": build_result.record}
    dataset_manifest = read_json(paths.dataset_manifest(
        (protocol.get("data") or {}).get("evaluation_split", "postselection_test")))
    validation = protocol.get("validation") or {}
    spec = verify.EvaluationSpec(
        kernel_dir=kernel_dir, driver=build_result.binary,
        build_record=build_result.record, backend=backend, shape=shape,
        split=(protocol.get("data") or {}).get("evaluation_split", "postselection_test"),
        atol=validation.get("atol", criteria.ATOL),
        rtol=validation.get("rtol", criteria.RTOL),
        relative_l2=validation.get("relative_l2", criteria.RELATIVE_L2_TOLERANCE),
        strict_abs=validation.get("strict_abs", criteria.STRICT_ABS_TOLERANCE),
        timeout_seconds=float(validation.get("timeout_seconds", 1800.0)),
        repeat_checks=int(validation.get("repeat_checks", 8)),
        hypotheses_max_shape=int(validation.get("hypotheses_max_shape", 512)),
        device=device, task_id=label, manifest=dataset_manifest)
    cache = reference.ReferenceCache(paths.references)
    record = verify.evaluate_kernel(spec, cache, scratch_dir=evidence_dir / "scratch")
    verify.write_validation(evidence_dir, record,
                            validation_csv=evidence_dir / "validation.csv",
                            coefficient_csv=evidence_dir / "coefficient_checks.csv",
                            validation_json=evidence_dir / "validation.json")
    timing_spec = timing.TimingSpec(label=label, driver=build_result.binary,
                                    build_record=build_result.record, shape=shape,
                                    backend=backend, task_id=label, device=device)
    timing_record = timing.time_kernel(timing_spec, evidence_dir / "timing")
    timing.write_timing(evidence_dir, timing_record,
                        timing_csv=evidence_dir / "timing.csv",
                        timing_json=evidence_dir / "timing.json")
    return {"status": record["summary"]["verdict"],
            "validation": record["summary"],
            "timing": timing_record["summary"],
            "binary_sha256": build_result.record.get("binary_sha256")}


def run_alias_without_parameter_search(paths: CampaignPaths, protocol: dict[str, Any],
                                       parent_id: str, *, backend: str,
                                       device: int = 0) -> dict[str, Any]:
    directory, record = _prepare_branch(paths, parent_id, "without_parameter_search")
    if record.get("status") not in ("started",):
        return record
    parent_status = (read_json(paths.task_status(parent_id))
                     if paths.task_status(parent_id).exists() else {})
    if parent_status.get("status") in ("no_candidate", "generation_failed"):
        record.update(status="parent_failed", reason="parent produced no delivery",
                      parent_status=parent_status.get("status"))
        write_json(directory / "branch.json", record)
        return record
    checkpoint = _parent_checkpoint(paths, parent_id, "p2")
    if checkpoint is None:
        record.update(status="parent_failed",
                      reason="parent has no P2 checkpoint to alias")
        write_json(directory / "branch.json", record)
        return record
    kernel_dir = _kernel_from_checkpoint(paths, directory, checkpoint)
    result = _validate_and_time(paths, protocol, kernel_dir, backend=backend,
                                shape=int(parent_status.get("shape")
                                          or checkpoint.get("selection", {}).get("shape", {}).get("M")
                                          or 0),
                                label=f"{parent_id}:woP3", out_dir=directory,
                                device=device, evidence_dir=directory)
    record.update(status="completed", alias_of=checkpoint.get("artifact_hash"),
                  kernel=checkpoint.get("kernel", {}).get("files"),
                  result=result, finished_at=utc_now(),
                  note="same P2 seed, independently timed; P3 contribution is the "
                       "paired difference")
    write_json(directory / "branch.json", record)
    return record


def run_without_feedback(paths: CampaignPaths, protocol: dict[str, Any],
                         parent_id: str, *, backend: str, device: int = 0,
                         workspace: isolation.Workspace | None = None,
                         config_path: Path | None = None) -> dict[str, Any]:
    directory, record = _prepare_branch(paths, parent_id, "without_feedback")
    if record.get("status") not in ("started",):
        return record
    parent_status = (read_json(paths.task_status(parent_id))
                     if paths.task_status(parent_id).exists() else {})
    if parent_status.get("status") in ("no_candidate", "generation_failed"):
        record.update(status="parent_failed", reason="parent produced no P1 checkpoint",
                      parent_status=parent_status.get("status"))
        write_json(directory / "branch.json", record)
        return record
    checkpoint = _parent_checkpoint(paths, parent_id, "p1")
    if checkpoint is None:
        record.update(status="parent_failed",
                      reason="parent has no P1 checkpoint; the branch stays in the "
                             "40-parent denominator")
        write_json(directory / "branch.json", record)
        return record

    shape = int(checkpoint.get("selection", {}).get("shape", {}).get("M")
                or parent_status.get("shape") or 0)
    if workspace is None:
        workspace = isolation.create_workspace(paths, f"{parent_id}__woFB")
    seed_dir = ensure_dir(workspace.package_dir / "gemm_code" / "code" / "p1_seed")
    for name in (checkpoint.get("kernel", {}).get("files") or {}):
        source = Path(checkpoint["kernel"]["source_dir"]) / name
        if source.is_file():
            (seed_dir / name).write_bytes(source.read_bytes())

    budget = protocol.get("budget") or {}
    search = budget.get("search") or {}
    config = (config_path or (workspace.package_dir / "conf.yaml"))
    if not Path(config).exists():
        base = config_module.load_yaml(
            Path((protocol.get("source") or {}).get("base_config_path")))
        config.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")
    argv = [
        "-m", "SCOPE.tune",
        "--backend", backend,
        "--source", str(seed_dir),
        "--matrix-size", str(shape), str(shape), str(shape),
        "--config", str(config),
        "--output", str(workspace.package_dir / "results" / "branch_tuning"),
        "--max-candidates", str(search.get("max_candidates_per_shape") or 0),
        "--rounds", str(search.get("rounds") or 3),
        "--aggregation", "trimmed_mean",
        "--benchmark-runs", str(search.get("screening_benchmark_runs") or 1),
        "--warmup-runs", "0",
        "--build-platform", str((protocol.get("environment") or {}).get(
            "toolchain_driver", {}).get("build_platform") or "windows"),
    ]
    env = workspace.env()
    result = run_command([__import__("sys").executable, *argv],
                         cwd=workspace.root, env=env,
                         timeout=float(budget.get("per_task_wall_seconds") or 0) or None)
    record.update(seed_checkpoint=checkpoint.get("artifact_hash"),
                  tune_argv=argv, tune_returncode=result["returncode"],
                  tune_timed_out=result["timed_out"],
                  tune_stdout_tail=(result["stdout"] or "")[-2000:],
                  tune_stderr_tail=(result["stderr"] or "")[-2000:])
    if result["timed_out"] or result["returncode"] != 0:
        record.update(status="branch_failed",
                      category=("budget_exhausted" if result["timed_out"]
                                else "branch_runner_error"))
        write_json(directory / "branch.json", record)
        return record

    summary_path = None
    for candidate in sorted((workspace.package_dir / "results" / "branch_tuning").rglob(
            "summary.json")):
        summary_path = candidate
        break
    if summary_path is None:
        record.update(status="branch_failed", category="no_tuning_summary")
        write_json(directory / "branch.json", record)
        return record
    summary = read_json(summary_path)
    winner = None
    for item in summary.get("shapes") or []:
        if int((item.get("shape") or {}).get("M") or 0) == shape and \
                item.get("status") == "completed":
            winner = item
            break
    if winner is None:
        record.update(status="branch_failed", category="no_verified_result",
                      tuning_summary=str(summary_path))
        write_json(directory / "branch.json", record)
        return record

    kernel_dir = Path(str(winner.get("final_code_dir")))
    if not kernel_dir.is_absolute():
        kernel_dir = workspace.package_dir / kernel_dir
    evidence = _validate_and_time(paths, protocol, kernel_dir, backend=backend,
                                  shape=shape, label=f"{parent_id}:woFB",
                                  out_dir=directory, device=device,
                                  evidence_dir=directory)
    record.update(status="completed", tuning_summary=str(summary_path),
                  search_cost=winner.get("search_cost"),
                  stop_reason=winner.get("stop_reason"),
                  result=evidence, finished_at=utc_now(),
                  note="P2 skipped; the same P3 protocol ran from the parent's P1 "
                       "checkpoint, so the difference is the feedback stage")
    write_json(directory / "branch.json", record)
    if workspace is not None:
        isolation.cleanup_workspace(workspace)
    return record


def enumerate_branch_plan(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for record in (protocol.get("tasks") or {}).get("task_records") or []:
        if record.get("block") != "E1" or record.get("kind") != "branch":
            continue
        plan.append({"kind": "without_feedback", "parent_task_id": record["parent_task_id"],
                     "branch_task_id": record["task_id"], "shape": record["shape_M"]})
    for record in (protocol.get("tasks") or {}).get("task_records") or []:
        if record.get("block") != "E1" or record.get("kind") != "alias":
            continue
        plan.append({"kind": "without_parameter_search",
                     "parent_task_id": record["parent_task_id"],
                     "branch_task_id": record["task_id"], "shape": record["shape_M"]})
    return plan


def branch_summary(paths: CampaignPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths.branches.rglob("branch.json")):
        record = read_json(path)
        result = record.get("result") or {}
        rows.append({
            "branch_task_id": path.parent.name,
            "kind": record.get("kind"),
            "parent_task_id": record.get("parent_task_id"),
            "status": record.get("status"),
            "independent_n": record.get("independent_n"),
            "validation_verdict": (result.get("validation") or {}).get("verdict"),
            "gflops_mean": (result.get("timing") or {}).get("gflops_mean"),
            "binary_sha256": result.get("binary_sha256"),
            "parent_checkpoint": record.get("seed_checkpoint") or record.get("alias_of"),
        })
    return rows
