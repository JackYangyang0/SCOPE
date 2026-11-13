
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import criteria, data, reference as reference_module
from .cases import (case_file_name, parse_driver_stdout, read_output_file,
                    read_case_file, write_case_file)
from .errors import InfraError
from .util import (ensure_dir, pinned_thread_env, run_command, sha256_bytes, sha256_file,
                   sha256_value, utc_now, write_csv, write_json)

VALIDATION_VERSION = "scope-exp-validation-v1"


@dataclass
class EvaluationSpec:

    kernel_dir: Path
    driver: Path
    build_record: dict[str, Any]
    backend: str
    shape: int
    split: str
    atol: float = criteria.ATOL
    rtol: float = criteria.RTOL
    relative_l2: float = criteria.RELATIVE_L2_TOLERANCE
    strict_abs: float = criteria.STRICT_ABS_TOLERANCE
    timeout_seconds: float = 600.0
    repeat_checks: int = 8
    hypotheses_max_shape: int = 512
    include_coefficients: bool = True
    device: int = 0
    keep_outputs: bool = False
    keep_cases: bool = False
    cleanup_scratch: bool = True
    task_id: str = ""
    manifest: dict[str, Any] | None = None
    legacy_mkn_order: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "validation_version": VALIDATION_VERSION,
            "kernel_dir": str(self.kernel_dir),
            "driver": str(self.driver),
            "driver_sha256": sha256_file(self.driver) if Path(self.driver).exists() else None,
            "backend": self.backend,
            "shape": self.shape,
            "split": self.split,
            "atol": self.atol,
            "rtol": self.rtol,
            "relative_l2": self.relative_l2,
            "strict_abs": self.strict_abs,
            "timeout_seconds": self.timeout_seconds,
            "repeat_checks": self.repeat_checks,
            "hypotheses_max_shape": self.hypotheses_max_shape,
            "include_coefficients": self.include_coefficients,
            "device": self.device,
            "task_id": self.task_id,
            "legacy_mkn_order": self.legacy_mkn_order,
            "binary_sha256": self.build_record.get("binary_sha256"),
        }


@dataclass
class CaseOutcome:
    row: dict[str, Any] = field(default_factory=dict)
    output: np.ndarray | None = None
    infrastructure_failure: bool = False


def run_case_driver(driver: Path, case_path: Path, output_path: Path, *,
                    mode: str = "compute", guard: bool = True,
                    timeout: float = 600.0, warmup_launches: int = 5,
                    iters: int = 50, device: int = 0, backend: str = "cpu",
                    legacy_mkn_order: bool = False,
                    cwd: Path | None = None) -> dict[str, Any]:
    argv = [str(driver), "--case", str(case_path), "--mode", mode]
    if mode == "compute":
        argv += ["--out", str(output_path)]
    else:
        argv += ["--warmup-launches", str(warmup_launches), "--iters", str(iters)]
        if backend == "cuda":
            argv += ["--device", str(device)]
    if guard and mode == "compute":
        argv.append("--guard")
    if legacy_mkn_order:
        argv.append("--legacy-mkn-order")
    result = run_command(argv, cwd=cwd, timeout=timeout,
                         env=pinned_thread_env())
    parsed = parse_driver_stdout(result["stdout"])
    result["driver_record"] = parsed.get("record") if parsed["found"] else None
    result["parse_error"] = parsed.get("error")
    return result


def _classify_run_failure(run: dict[str, Any]) -> tuple[str, bool]:
    if run.get("timed_out"):
        return "timeout", False
    code = run.get("returncode")
    if code == 5:
        return "out_of_bounds_write", False
    if code == 4:
        return "kernel_error", False
    if code == 3:
        return "driver_io_error", True
    if code is None:
        return "driver_not_started", True
    if code != 0:
        return f"driver_exit_{code}", False
    if not run.get("driver_record"):
        return "driver_no_record", False
    return "ok", False


def _alternatives(case: dict[str, Any], AB: np.ndarray | None) -> dict[str, np.ndarray]:
    alpha = float(case["alpha"])
    beta = float(case["beta"])
    A = np.asarray(case["A"], dtype=np.float64)
    B = np.asarray(case["B"], dtype=np.float64)
    C0 = np.asarray(case["C0"], dtype=np.float64)
    hypotheses: dict[str, np.ndarray] = {}
    if AB is None:
        return hypotheses
    hypotheses["reference"] = alpha * AB + beta * C0
    if A.shape[0] == B.shape[1]:
        hypotheses["transposed_output"] = alpha * AB.T + beta * C0
    hypotheses["c0_leaked_beta_one"] = alpha * AB + C0
    hypotheses["c0_ignored_beta_zero"] = alpha * AB
    hypotheses["alpha_double_applied"] = alpha * alpha * AB + beta * C0
    hypotheses["output_all_zero"] = np.zeros_like(AB)
    K = A.shape[1]
    if K > 1:
        short = A[:, :K - 1] @ B[:K - 1, :]
        hypotheses["missing_last_k_term"] = alpha * short + beta * C0
    if A.shape[0] == A.shape[1]:
        hypotheses["row_major_a_read_as_column_major"] = alpha * (A.T @ B) + beta * C0
    return hypotheses


def evaluate_kernel(spec: EvaluationSpec, reference_cache: reference_module.ReferenceCache | None = None,
                    *, progress: Any = None, scratch_dir: Path | None = None,
                    entries: Sequence[dict[str, Any]] | None = None,
                    ) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    spec.kernel_dir = Path(spec.kernel_dir).resolve()
    spec.driver = Path(spec.driver).resolve()
    if not spec.driver.exists():
        raise InfraError(f"evaluation driver missing: {spec.driver}", stage="evaluate")

    manifest = spec.manifest or data.build_manifest([spec.shape], split=spec.split,
                                                    include_coefficients=spec.include_coefficients)
    manifest_hash = manifest.get("manifest_hash") or sha256_value(manifest)
    if entries is None:
        entries = [entry for entry in manifest["entries"]
                   if int(entry["shape"]["M"]) == int(spec.shape)]

    reference_cache = reference_cache or reference_module.ReferenceCache(
        Path(scratch_dir or spec.kernel_dir) / "references",
        kind="cpu_blas_fp64", store_arrays=True)
    ensure_dir(Path(scratch_dir or spec.kernel_dir))

    rows: list[dict[str, Any]] = []
    hypotheses_enabled = int(spec.shape) <= int(spec.hypotheses_max_shape)
    repeat_budget = int(spec.repeat_checks)
    repeated = 0

    for index, entry in enumerate(entries):
        row, outcome = _evaluate_one(entry, spec, reference_cache, hypotheses_enabled,
                                     scratch_dir, repeat=(repeated < repeat_budget))
        if row["repeat_checked"]:
            repeated += 1
        rows.append(row)
        if progress is not None:
            progress(index + 1, len(entries), row)

    summary = _summarize(rows, spec, manifest_hash)
    if spec.cleanup_scratch and scratch_dir is not None:
        cleanup_cases(Path(scratch_dir) / "eval")
    record = {
        "validation_version": VALIDATION_VERSION,
        "task_id": spec.task_id,
        "spec": spec.as_record(),
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "manifest_hash": manifest_hash,
        "policies": [policy.as_record() for policy in criteria.ALL_POLICIES],
        "summary": summary,
        "rows": rows,
    }
    return record


def cleanup_cases(directory: Path) -> int:
    removed = 0
    if not Path(directory).exists():
        return 0
    for pattern in ("*.case", "*.out", "*.repeat.out", "*.tmp"):
        for path in Path(directory).glob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _evaluate_one(entry: dict[str, Any], spec: EvaluationSpec,
                  reference_cache: reference_module.ReferenceCache,
                  hypotheses_enabled: bool, scratch_dir: Path | None,
                  *, repeat: bool) -> tuple[dict[str, Any], CaseOutcome]:
    group = entry.get("group", "main")
    coefficients = {"name": entry["coefficient_set"], "alpha": entry["alpha"],
                    "beta": entry["beta"]}
    case = data.make_case(entry["case_id"], entry["shape"]["M"], entry["shape"]["N"],
                          entry["shape"]["K"], c0_variant=entry["c0_variant"],
                          split=entry["split"], coefficients=coefficients)
    base_row: dict[str, Any] = {
        "group": group,
        "case_id": entry["case_id"],
        "family": entry["family"],
        "compat_only": entry["compat_only"],
        "c0_variant": entry["c0_variant"],
        "alpha": entry["alpha"],
        "beta": entry["beta"],
        "coefficient_set": entry["coefficient_set"],
        "shape": f'{entry["shape"]["M"]}x{entry["shape"]["N"]}x{entry["shape"]["K"]}',
        "split": entry["split"],
        "case_key": entry["key"],
        "case_sha256": sha256_value(case["hashes"]),
        "manifest_hashes_match": case["hashes"] == entry["hashes"],
        "repeat_checked": False,
        "repeat_max_abs_diff": None,
        "status": "not_run",
        "main_policy": "not_run",
        "online_gate": "not_run",
        "strict_abs": "not_run",
        "reference_valid": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "relative_l2_error": None,
        "elementwise_bad_count": None,
        "non_finite_count": None,
        "bad_positions": None,
        "strict_abs_pass": None,
        "output_sha256": None,
        "reference_sha256": None,
        "diagnosis": None,
        "driver_returncode": None,
        "driver_timed_out": False,
        "runtime_seconds": None,
        "notes": [],
    }

    if not base_row["manifest_hashes_match"]:
        base_row.update(status="input_byte_drift", main_policy="not_run",
                        notes=["regenerated input bytes differ from the frozen manifest"])
        return base_row, CaseOutcome(base_row, infrastructure_failure=True)

    work = Path(scratch_dir or spec.kernel_dir) / "eval"
    ensure_dir(work)
    case_path = work / case_file_name(entry["key"])
    if not case_path.exists():
        write_case_file(case_path, case)
    else:
        reread = read_case_file(case_path)
        if reread["sha256"] != sha256_value(case["hashes"]):
            write_case_file(case_path, case)

    output_path = work / (case_path.stem + ".out")
    run = run_case_driver(spec.driver, case_path, output_path, timeout=spec.timeout_seconds,
                          device=spec.device, backend=spec.backend,
                          legacy_mkn_order=spec.legacy_mkn_order)
    category, infra = _classify_run_failure(run)
    base_row["driver_returncode"] = run["returncode"]
    base_row["driver_timed_out"] = bool(run.get("timed_out"))
    base_row["runtime_seconds"] = run.get("duration_seconds")

    if category != "ok":
        base_row["status"] = category
        base_row["main_policy"] = "fail" if category != "driver_io_error" else "not_run"
        base_row["notes"].append((run.get("parse_error") or "")[:200])
        if run.get("stderr"):
            base_row["notes"].append(run["stderr"][-300:].replace("\n", " | "))
        if output_path.exists():
            output_path.unlink()
        return base_row, CaseOutcome(base_row, infrastructure_failure=infra)

    record = run["driver_record"] or {}
    guard_corrupted = bool(record.get("guard_corrupted"))
    if guard_corrupted:
        base_row["status"] = "out_of_bounds_write"
        base_row["main_policy"] = "fail"
        base_row["notes"].append("guard band around C was modified")
        if output_path.exists():
            output_path.unlink()
        return base_row, CaseOutcome(base_row)

    M, N = entry["shape"]["M"], entry["shape"]["N"]
    try:
        actual_f32 = read_output_file(output_path, M, N)
    except (ValueError, OSError) as exc:
        base_row["status"] = "output_missing_or_short"
        base_row["main_policy"] = "fail"
        base_row["notes"].append(str(exc)[:300])
        return base_row, CaseOutcome(base_row)
    finally:
        if output_path.exists() and not spec.keep_outputs:
            output_path.unlink()

    reference = reference_cache.get(case, with_ab=hypotheses_enabled)
    metrics = criteria.output_metrics(actual_f32.astype(np.float64), reference.C)
    policies = criteria.all_policies(metrics)
    main = policies[criteria.POLICY_MAIN]
    online = policies[criteria.POLICY_ONLINE]
    strict = policies[criteria.POLICY_STRICT]

    base_row.update(
        status="pass" if main["passed"] else "fail",
        main_policy=main["status"],
        online_gate=online["status"],
        strict_abs=strict["status"],
        strict_abs_pass=strict["passed"],
        max_abs_error=metrics["max_abs_error"],
        max_rel_error=metrics["max_rel_error"],
        relative_l2_error=metrics["relative_l2_error"],
        elementwise_bad_count=metrics["elementwise_bad_count"],
        non_finite_count=metrics["non_finite_count"],
        bad_positions=[int(p) for p in metrics["elementwise_bad_positions_flat"]],
        output_sha256=sha256_bytes(actual_f32.tobytes(order="C")),
        reference_sha256=reference.sha256,
        reference_valid=bool(reference.meta.get("valid")),
        notes=list(main["reasons"])[:4],
    )

    if not main["passed"]:
        if metrics["non_finite_count"]:
            base_row["diagnosis"] = {"primary": "non_finite_output",
                                     "evidence": {"count": metrics["non_finite_count"]}}
        elif hypotheses_enabled:
            diagnosis = criteria.diagnose_mismatch(
                actual_f32.astype(np.float64),
                _alternatives(case, reference.AB))
            base_row["diagnosis"] = diagnosis.as_record()
        else:
            base_row["diagnosis"] = {"primary": "unexplained_large_shape",
                                     "evidence": {"note": "hypotheses disabled above "
                                                          f"{spec.hypotheses_max_shape}"}}

    if repeat:
        second_path = work / (case_path.stem + ".repeat.out")
        repeat_run = run_case_driver(spec.driver, case_path, second_path,
                                     timeout=spec.timeout_seconds, device=spec.device,
                                     backend=spec.backend)
        base_row["repeat_checked"] = True
        if repeat_run.get("returncode") == 0 and second_path.exists():
            second = read_output_file(second_path, M, N).astype(np.float64)
            difference = float(np.max(np.abs(second - actual_f32.astype(np.float64))))
            if np.isfinite(difference):
                base_row["repeat_max_abs_diff"] = difference
            else:
                base_row["notes"].append("repeat difference is non-finite")
            second_path.unlink()
        else:
            base_row["notes"].append("repeat run did not complete")

    return base_row, CaseOutcome(base_row, output=actual_f32)


def _summarize(rows: Sequence[dict[str, Any]], spec: EvaluationSpec,
               manifest_hash: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"main": [], "coefficients": []}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)

    def _group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        planned = len(items)
        passed = [row for row in items if row["main_policy"] == "pass"]
        failed = [row for row in items if row["main_policy"] == "fail"]
        not_run = [row for row in items if row["main_policy"] == "not_run"]
        infra = [row for row in items if row["status"] in
                 ("input_byte_drift", "driver_io_error", "driver_not_started")]
        errors: dict[str, int] = {}
        for row in items:
            if row["main_policy"] != "pass":
                errors[row["status"]] = errors.get(row["status"], 0) + 1
        by_family: dict[str, dict[str, int]] = {}
        for row in items:
            entry = by_family.setdefault(row["family"], {"planned": 0, "passed": 0})
            entry["planned"] += 1
            entry["passed"] += int(row["main_policy"] == "pass")
        strict_passes = sum(1 for row in items if row["strict_abs_pass"] is True)
        repeat_values = [row["repeat_max_abs_diff"] for row in items
                         if row["repeat_max_abs_diff"] is not None]
        return {
            "planned": planned,
            "executed": planned - len(not_run),
            "passed": len(passed),
            "failed": len(failed),
            "not_run": len(not_run),
            "infrastructure_incomplete": len(infra),
            "error_taxonomy": dict(sorted(errors.items())),
            "by_family": dict(sorted(by_family.items())),
            "strict_abs_pass_count": strict_passes,
            "compat_only": {
                "planned": sum(1 for row in items if row["compat_only"]),
                "passed": sum(1 for row in items if row["compat_only"] and row["main_policy"] == "pass"),
            },
            "repeat_checks": {
                "count": len(repeat_values),
                "max_abs_diff": max(repeat_values) if repeat_values else None,
            },
        }

    main_summary = _group_summary(groups.get("main", []))
    coefficient_summary = _group_summary(groups.get("coefficients", []))
    reference_invalid = [row for row in rows if row["reference_valid"] is False]
    repeat_values = [row["repeat_max_abs_diff"] for row in rows
                     if row["repeat_max_abs_diff"] is not None]
    nondeterministic = [row for row in rows
                        if row["repeat_max_abs_diff"] not in (None, 0.0)]

    if main_summary["planned"] == 0:
        verdict = "not_run"
    elif reference_invalid:
        verdict = "reference_invalid"
    elif main_summary["not_run"] or main_summary["infrastructure_incomplete"]:
        verdict = "incomplete"
    elif main_summary["passed"] == main_summary["planned"]:
        verdict = "pass"
    else:
        verdict = "fail"

    coefficient_verdict = ("not_run" if coefficient_summary["planned"] == 0 else
                           "pass" if coefficient_summary["passed"] == coefficient_summary["planned"]
                           else "incomplete" if coefficient_summary["not_run"]
                           else "fail")

    return {
        "manifest_hash": manifest_hash,
        "main": main_summary,
        "coefficients": coefficient_summary,
        "verdict": verdict,
        "main_verdict": verdict,
        "coefficient_verdict": coefficient_verdict,
        "delivery": verdict == "pass",
        "reference_invalid_cases": [row["case_key"] for row in reference_invalid],
        "determinism": {
            "repeats": len(repeat_values),
            "max_abs_diff": max(repeat_values) if repeat_values else None,
            "nondeterministic_cases": [row["case_key"] for row in nondeterministic],
        },
        "strict_abs_total_pass": main_summary["strict_abs_pass_count"],
        "notes": [
            "main policy is the delivery criterion; online_gate is the generation "
            "gate replayed for comparison only; strict_abs is diagnostic",
            "a case that fails is never retried or replaced",
        ],
    }


def write_validation(paths_root: Path, record: dict[str, Any], *,
                     validation_csv: Path, coefficient_csv: Path,
                     validation_json: Path | None = None) -> dict[str, Path]:
    rows = record["rows"]
    main_rows = [row for row in rows if row["group"] == "main"]
    coefficient_rows = [row for row in rows if row["group"] == "coefficients"]
    columns = ["group", "case_id", "family", "compat_only", "c0_variant",
               "alpha", "beta", "coefficient_set", "shape", "split",
               "case_key", "case_sha256", "status", "main_policy", "online_gate",
               "strict_abs", "strict_abs_pass", "max_abs_error", "max_rel_error",
               "relative_l2_error", "elementwise_bad_count", "non_finite_count",
               "reference_valid", "reference_sha256", "output_sha256",
               "repeat_checked", "repeat_max_abs_diff", "driver_returncode",
               "driver_timed_out", "runtime_seconds", "diagnosis", "notes",
               "manifest_hashes_match", "bad_positions"]
    written = {
        "validation_csv": write_csv(validation_csv, main_rows, columns),
        "coefficient_csv": write_csv(coefficient_csv, coefficient_rows, columns),
    }
    if validation_json is not None:
        written["validation_json"] = write_json(validation_json, record)
    write_json(Path(paths_root) / "validation.summary.json", record["summary"])
    return written
