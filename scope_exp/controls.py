
from __future__ import annotations

import shutil
from dataclasses import dataclass

from . import cases, sanitizer as sanitizer_module
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import HARNESS_VERSION
from . import build as build_module
from . import data, reference as reference_module, verify
from .errors import GateError, InfraError
from .paths import CampaignPaths, repo_root
from .util import ensure_dir, read_json, sha256_file, sha256_value, utc_now, write_json

CONTROLS_VERSION = "scope-exp-controls-v1"
CONTROL_SHAPE = 64
NON_SQUARE_SHAPES = ((127, 129, 65), (65, 127, 129))
TRIGGER_CASE = "uniform_small.s00"

DRIVERS = Path(__file__).resolve().parent / "drivers"
CONTROL_SOURCES = DRIVERS / "controls"


@dataclass
class ControlCase:
    name: str
    backend: str
    source: str
    purpose: str
    expect_main: str
    expect_coefficients: str
    expect_trigger_fail: bool = False
    expect_status: Sequence[str] = ()
    expect_pass_cases: Sequence[str] = ()
    notes: str = ""

    def as_record(self) -> dict[str, Any]:
        return {"name": self.name, "backend": self.backend, "source": self.source,
                "purpose": self.purpose, "expect_main": self.expect_main,
                "expect_coefficients": self.expect_coefficients,
                "expect_trigger_fail": self.expect_trigger_fail,
                "expect_status": list(self.expect_status),
                "expect_pass_cases": list(self.expect_pass_cases),
                "notes": self.notes}


CPU_CONTROLS: tuple[ControlCase, ...] = (
    ControlCase("reference_fp64", "cpu", "cpu_reference_fp64.c",
                "positive: FP64 accumulation with one float32 rounding",
                "pass", "pass"),
    ControlCase("all_zero", "cpu", "cpu_all_zero.c",
                "negative: zero output, ignores inputs (must fail on non-zero expectation)",
                "fail", "fail", expect_trigger_fail=True,
                expect_pass_cases=("zero",),
                notes="passes the zero configuration, so a single case would be insufficient"),
    ControlCase("partial_write", "cpu", "cpu_partial_write.c",
                "negative: writes only the first half of the rows",
                "fail", "fail", expect_trigger_fail=True),
    ControlCase("wrong_layout", "cpu", "cpu_wrong_layout.c",
                "negative: reads A as column-major",
                "fail", "fail", expect_trigger_fail=True),
    ControlCase("short_k", "cpu", "cpu_short_k.c",
                "negative: drops the last k term",
                "fail", "fail", expect_trigger_fail=True),
    ControlCase("reads_c0", "cpu", "cpu_reads_c0.c",
                "negative: applies beta = 1 regardless of the contract",
                "fail", "fail", expect_trigger_fail=True),
    ControlCase("nan_output", "cpu", "cpu_nan_output.c",
                "negative: NaN output must be classified as non-finite",
                "fail", "fail", expect_trigger_fail=True,
                expect_status=("fail",)),
    ControlCase("oob_write", "cpu", "cpu_oob_write.c",
                "negative: writes one element past C; the guard band must catch it",
                "fail", "fail", expect_trigger_fail=True,
                expect_status=("out_of_bounds_write",)),
    ControlCase("stale_accumulator", "cpu", "cpu_stale_accumulator.c",
                "negative: accumulator not reset across columns",
                "fail", "fail", expect_trigger_fail=True),
    ControlCase("off_by_one_read", "cpu", "cpu_off_by_one_read.c",
                "negative: k index of A shifted by one",
                "fail", "fail", expect_trigger_fail=True),
    ControlCase("drops_alpha", "cpu", "cpu_drops_alpha.c",
                "negative: ignores alpha; only the coefficient group can detect this",
                "pass", "fail"),
    ControlCase("drops_beta", "cpu", "cpu_drops_beta.c",
                "negative: ignores beta; only the coefficient group can detect this",
                "pass", "fail"),
)

CUDA_CONTROLS: tuple[ControlCase, ...] = (
    ControlCase("cuda_reference_double", "cuda", "cuda_reference_double.cuh",
                "positive: one output per thread, double accumulation",
                "pass", "pass"),
    ControlCase("cuda_all_zero", "cuda", "cuda_all_zero.cuh",
                "negative: zero output", "fail", "fail", expect_trigger_fail=True),
    ControlCase("cuda_short_k", "cuda", "cuda_short_k.cuh",
                "negative: drops the last k term", "fail", "fail",
                expect_trigger_fail=True),
    ControlCase("cuda_oob_write", "cuda", "cuda_oob_write.cuh",
                "negative: out-of-bounds write; memcheck must report it",
                "fail", "fail", expect_trigger_fail=True,
                notes="needs compute-sanitizer; the host guard may not see it"),
    ControlCase("cuda_shared_no_sync", "cuda", "cuda_shared_no_sync.cuh",
                "negative: shared-memory race with no __syncthreads",
                "fail", "fail", expect_trigger_fail=True,
                notes="applicability probe for racecheck/synccheck"),
)


def controls_for(backend: str) -> tuple[ControlCase, ...]:
    if backend == "cuda":
        return CUDA_CONTROLS
    return CPU_CONTROLS


def prepare_control_kernel(source_name: str, backend: str, scratch: Path) -> Path:
    source = CONTROL_SOURCES / source_name
    if not source.exists():
        raise InfraError(f"control source missing: {source}", stage="controls")
    kernel_dir = ensure_dir(Path(scratch) / "kernel")
    for existing in kernel_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    if backend == "cuda":
        shutil.copyfile(source, kernel_dir / "cuda_kernel.cuh")
    else:
        shutil.copyfile(source, kernel_dir / "cpu_kernel.c")
        skeleton_header = repo_root() / "gemm_code" / "cpu_skeleton" / "kernel.h"
        if not skeleton_header.exists():
            raise InfraError(f"CPU kernel header missing: {skeleton_header}",
                             stage="controls")
        shutil.copyfile(skeleton_header, kernel_dir / "kernel.h")
    return kernel_dir


def restricted_manifest(shape: int, split: str, include_coefficients: bool) -> dict[str, Any]:
    return data.build_manifest([shape], split=split,
                               include_coefficients=include_coefficients)


SANITIZER_CONTROLS = {"cuda_oob_write": "memcheck"}


def _sanitizer_hazard(binary, shape: int, scratch: Path, tool: str,
                      timeout_seconds: float, device: int = 0) -> dict:
    case = data.make_case("uniform_small.s00", shape, shape, shape,
                          c0_variant="scalar", split="development")
    out_dir = ensure_dir(Path(scratch) / "sanitizer")
    case_path = out_dir / "trigger.case"
    cases.write_case_file(case_path, case)
    return sanitizer_module.run_sanitizer(
        Path(binary), case_path, tool=tool, timeout_seconds=timeout_seconds,
        out_dir=out_dir, backend="cuda", device=device)


def run_control(control: ControlCase, scratch: Path, *, shape: int, split: str,
                reference_cache: reference_module.ReferenceCache,
                toolchain: build_module.Toolchain,
                include_coefficients: bool = True,
                timeout_seconds: float = 600.0) -> dict[str, Any]:
    kernel_dir = prepare_control_kernel(control.source, control.backend, scratch)
    build_result = build_module.build_eval_driver(
        kernel_dir, Path(scratch) / "build", backend=control.backend,
        toolchain=toolchain, timeout=timeout_seconds)
    if not build_result.ok:
        return {"control": control.as_record(), "ok": False,
                "reason": f"control kernel did not build: {build_result.record.get('error')}",
                "build": build_result.record}

    manifest = restricted_manifest(shape, split, include_coefficients)
    spec = verify.EvaluationSpec(
        kernel_dir=kernel_dir, driver=build_result.binary,
        build_record=build_result.record, backend=control.backend,
        shape=shape, split=split, timeout_seconds=timeout_seconds,
        repeat_checks=1, hypotheses_max_shape=shape,
        include_coefficients=include_coefficients, task_id=f"control:{control.name}",
        manifest=manifest)
    record = verify.evaluate_kernel(spec, reference_cache,
                                    scratch_dir=Path(scratch) / "eval")
    summary = record["summary"]
    rows = record["rows"]

    observed_main = ("pass" if summary["main"]["planned"] > 0 and
                     summary["main"]["passed"] == summary["main"]["planned"] else
                     "fail" if summary["main"]["failed"] else "not_run")
    observed_coefficients = (
        "pass" if summary["coefficients"]["planned"] and
        summary["coefficients"]["passed"] == summary["coefficients"]["planned"] else
        "fail" if summary["coefficients"]["failed"] else "not_run")

    problems: list[str] = []
    sanitizer_record = None
    if control.name in SANITIZER_CONTROLS:
        sanitizer_record = _sanitizer_hazard(
            build_result.binary, shape, scratch,
            SANITIZER_CONTROLS[control.name], timeout_seconds)
        if sanitizer_record.get("status") != "fail":
            problems.append(
                f"{control.name}: {SANITIZER_CONTROLS[control.name]} reported "
                f"{sanitizer_record.get('status')}, expected a hazard")
    sanitizer_only = control.name in SANITIZER_CONTROLS
    if not sanitizer_only and control.expect_main != "any" \
            and observed_main != control.expect_main:
        problems.append(f"main group observed {observed_main}, expected {control.expect_main}")
    if not sanitizer_only and include_coefficients \
            and control.expect_coefficients != "any" \
            and observed_coefficients != control.expect_coefficients:
        problems.append(f"coefficient group observed {observed_coefficients}, "
                        f"expected {control.expect_coefficients}")

    trigger_rows = [row for row in rows
                    if row["group"] == "main" and row["case_id"] == TRIGGER_CASE
                    and row["c0_variant"] == "pattern"]
    if control.expect_trigger_fail and not sanitizer_only:
        if not trigger_rows:
            problems.append(f"trigger case {TRIGGER_CASE} was not evaluated")
        elif trigger_rows[0]["main_policy"] != "fail":
            problems.append(f"trigger case {TRIGGER_CASE} did not fail")

    if control.expect_status:
        statuses = {row["status"] for row in rows if row["group"] == "main"}
        if not (statuses & set(control.expect_status)):
            problems.append(f"none of the expected statuses {list(control.expect_status)} "
                            f"appeared; observed {sorted(statuses)}")

    for pass_case in control.expect_pass_cases:
        matching = [row for row in rows if row["group"] == "main"
                    and row["case_id"] == pass_case]
        if not matching:
            problems.append(f"expected-pass case {pass_case} was not evaluated")
        elif any(row["main_policy"] != "pass" for row in matching):
            problems.append(f"expected-pass case {pass_case} did not pass")

    return {
        "control": control.as_record(),
        "ok": not problems,
        "problems": problems,
        "observed_main": observed_main,
        "observed_coefficients": observed_coefficients,
        "trigger_case": TRIGGER_CASE if control.expect_trigger_fail else None,
        "summary": summary,
        "build": {"ok": build_result.ok,
                  "binary_sha256": build_result.record.get("binary_sha256"),
                  "argv": build_result.record.get("argv"),
                  "arch": build_result.record.get("arch")},
        "driver_sha256": sha256_file(build_result.binary),
        "kernel_dir": str(kernel_dir),
        "sanitizer": sanitizer_record,
    }


def reference_controls(shape: int = 40) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    for family in ("cancellation", "zero", "left_identity", "right_identity",
                   "impulse", "original_deterministic"):
        case = data.make_case(family, shape, shape, shape, split="development")
        reference = reference_module.build_reference(case)
        cross = reference_module.cross_check_reference(case, reference)
        valid = bool(reference.meta.get("valid"))
        ok = valid and cross["status"] == "agree"
        if not ok:
            failures.append(f"{family}: valid={valid} cross={cross['status']}")
        cases.append({"family": family, "reference_valid": valid,
                      "cross_check": cross, "ok": ok})
    random_case = data.make_case("uniform_wide.s03", shape, shape, shape,
                                split="development")
    reference = reference_module.build_reference(random_case)
    cross = reference_module.cross_check_reference(random_case, reference)
    ok = reference.meta["valid"] and cross["status"] == "agree"
    if not ok:
        failures.append(f"uniform_wide.s03: cross={cross['status']}")
    cases.append({"family": "uniform_wide.s03",
                  "reference_valid": reference.meta["valid"],
                  "cross_check": cross, "ok": ok})

    non_square: list[dict[str, Any]] = []
    for (M, N, K) in NON_SQUARE_SHAPES:
        for family in ("left_identity", "impulse", "cancellation", "zero"):
            case = data.make_case(family, M, N, K, split="development")
            reference = reference_module.build_reference(case)
            analytic = reference_module.analytic_output(case)
            difference = (float(np.max(np.abs(reference.C - analytic)))
                          if analytic is not None else None)
            allowed = 1e-9 * max(1.0, float(np.max(np.abs(reference.C))))
            ok = difference is not None and difference <= allowed
            if not ok:
                failures.append(f"{family} {M}x{N}x{K}: analytic difference {difference}")
            non_square.append({"family": family, "shape": [M, N, K],
                               "max_abs_difference": difference, "ok": ok})
    return {"ok": not failures, "problems": failures, "close_form": cases,
            "non_square": non_square}


def parameter_order_controls(shape_mnk: tuple[int, int, int] = NON_SQUARE_SHAPES[0],
                             *, scratch: Path, backend: str,
                             toolchain: build_module.Toolchain,
                             reference_cache: reference_module.ReferenceCache,
                             timeout_seconds: float = 600.0) -> dict[str, Any]:
    M, N, K = shape_mnk
    reference_source = ("cuda_reference_double.cuh" if backend == "cuda"
                        else "cpu_reference_fp64.c")
    kernel_dir = prepare_control_kernel(reference_source, backend,
                                        Path(scratch) / "order")
    build_result = build_module.build_eval_driver(
        kernel_dir, Path(scratch) / "order" / "build", backend=backend,
        toolchain=toolchain, timeout=timeout_seconds)
    if not build_result.ok:
        return {"ok": False, "reason": "reference kernel did not build",
                "problems": ["reference kernel did not build"],
                "build": build_result.record}

    case = data.make_case("uniform_small.s01", M, N, K, split="development")
    entry = data.manifest_entry(case, "main")
    manifest = {
        "generator_version": data.GENERATOR_VERSION,
        "split": "development",
        "shapes": [M, N, K],
        "entries": [entry],
        "manifest_hash": sha256_value([entry]),
    }
    results: dict[str, Any] = {}
    for label, legacy in (("declared_mnk_order", False), ("legacy_mkn_order", True)):
        spec = verify.EvaluationSpec(
            kernel_dir=kernel_dir, driver=build_result.binary,
            build_record=build_result.record, backend=backend, shape=M,
            split="development", timeout_seconds=timeout_seconds,
            repeat_checks=0, hypotheses_max_shape=max(M, N, K),
            include_coefficients=False, task_id=f"control:order:{label}",
            manifest=manifest, legacy_mkn_order=legacy)
        record = verify.evaluate_kernel(spec, reference_cache,
                                        scratch_dir=Path(scratch) / "order" / label,
                                        entries=[entry])
        results[label] = {"verdict": record["summary"]["verdict"],
                          "main": record["summary"]["main"],
                          "rows": [{k: row[k] for k in ("case_key", "status",
                                                        "main_policy", "max_abs_error")}
                                   for row in record["rows"]]}
    problems: list[str] = []
    if results["declared_mnk_order"]["verdict"] != "pass":
        problems.append("a correct kernel failed at the non-square shape; the "
                        "driver or reference is wrong")
    if results["legacy_mkn_order"]["verdict"] != "fail":
        problems.append("the legacy M,K,N positional order was not detected at a "
                        "non-square shape")
    return {"ok": not problems, "problems": problems,
            "shape_mnk": [M, N, K], "results": results,
            "note": "square shapes cannot demonstrate this; the control is "
                    "non-square by design"}


def run_controls(paths: CampaignPaths, *, backend: str = "cpu", shape: int = CONTROL_SHAPE,
                 split: str = "development", include_coefficients: bool = True,
                 timeout_seconds: float = 600.0,
                 protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_now()
    toolchain = build_module.discover_toolchain(backend)
    if not toolchain.available:
        record = {
            "controls_version": CONTROLS_VERSION, "ok": False, "backend": backend,
            "created_at": started_at, "toolchain": toolchain.as_record(),
            "reason": f"{backend} toolchain not available: {toolchain.extra}",
            "checks": [],
        }
        ensure_dir(paths.audit)
        write_json(paths.audit / "controls_gate.json", record)
        return record

    scratch_root = ensure_dir(paths.workspace_root / "controls" / backend)
    reference_cache = reference_module.ReferenceCache(paths.references,
                                                     kind="cpu_blas_fp64")
    checks: list[dict[str, Any]] = []
    for control in controls_for(backend):
        scratch = ensure_dir(scratch_root / control.name)
        checks.append(run_control(control, scratch, shape=shape, split=split,
                                  reference_cache=reference_cache,
                                  toolchain=toolchain,
                                  include_coefficients=include_coefficients,
                                  timeout_seconds=timeout_seconds))

    reference_record = reference_controls()
    order_record = parameter_order_controls(
        scratch=scratch_root, backend=backend, toolchain=toolchain,
        reference_cache=reference_cache, timeout_seconds=timeout_seconds)

    ok = (all(check["ok"] for check in checks)
          and reference_record["ok"] and order_record["ok"])
    driver_hashes = sorted({check.get("driver_sha256") for check in checks
                            if check.get("driver_sha256")})
    record = {
        "controls_version": CONTROLS_VERSION,
        "harness_version": HARNESS_VERSION,
        "created_at": started_at,
        "backend": backend,
        "shape": shape,
        "split": split,
        "include_coefficients": include_coefficients,
        "toolchain": toolchain.as_record(),
        "driver_sha256": driver_hashes,
        "checks": checks,
        "reference": reference_record,
        "parameter_order": order_record,
        "trigger_case": TRIGGER_CASE,
        "expectations": {control.name: control.as_record()
                         for control in controls_for(backend)},
        "ok": ok,
        "problems": ([f"{check['control']['name']}: {check['problems']}"
                      for check in checks if not check["ok"]]
                     + ([] if reference_record["ok"] else
                        [f"reference: {reference_record['problems']}"])
                     + ([] if order_record["ok"] else
                        [f"parameter_order: {order_record['problems']}"])),
        "protocol_hash": sha256_value(protocol) if protocol else None,
    }
    ensure_dir(paths.audit)
    write_json(paths.audit / "controls_gate.json", record)
    return record


def require_gate(paths: CampaignPaths, *, backend: str) -> dict[str, Any]:
    path = paths.audit / "controls_gate.json"
    if not path.exists():
        raise GateError(f"no control gate at {path}; run `scope-exp controls` first")
    record = read_json(path)
    if not record.get("ok"):
        raise GateError("control gate is red; fix the controls before running tasks")
    if record.get("backend") != backend:
        raise GateError(f"control gate was produced for backend "
                        f"{record.get('backend')!r}, but the campaign runs {backend!r}")
    return record


def gate_status(paths: CampaignPaths) -> dict[str, Any]:
    path = paths.audit / "controls_gate.json"
    if not path.exists():
        return {"present": False, "path": str(path)}
    record = read_json(path)
    return {"present": True, "ok": record.get("ok"), "backend": record.get("backend"),
            "created_at": record.get("created_at"),
            "failed_checks": [check["control"]["name"] for check in record.get("checks", [])
                              if not check.get("ok")],
            "problems": record.get("problems"), "path": str(path)}
