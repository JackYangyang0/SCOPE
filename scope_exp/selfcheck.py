from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from . import (HARNESS_VERSION, build, controls, data, isolation, reference,
               sanitizer, timing, util)
from .errors import InfraError
from .paths import CampaignPaths, repo_root
from .util import ensure_dir, run_command, sha256_value, utc_now, write_json

SELFCHECK_VERSION = "scope-exp-selfcheck-v1"


def _check(name: str, fn: Any) -> dict[str, Any]:
    try:
        result = fn()
        return {"check": name, "ok": bool(result.get("ok")), "details": result}
    except Exception as exc:
        return {"check": name, "ok": False,
                "details": {"error": f"{type(exc).__name__}: {exc}"}}


def check_package_import(paths: CampaignPaths) -> dict[str, Any]:
    workspace = isolation.create_workspace(paths, "selfcheck-import")
    try:
        result = run_command(
            [__import__("sys").executable, "-c",
             "import SCOPE.app as a; print('SCOPE.app', a.__file__); "
             "print('ROOT', a.ROOT)"],
            cwd=workspace.root, env=workspace.env(), timeout=300)
        return {
            "ok": result["returncode"] == 0,
            "cwd": str(workspace.root),
            "returncode": result["returncode"],
            "stdout": (result["stdout"] or "")[-800:],
            "stderr": (result["stderr"] or "")[-800:],
            "note": "the checkout directory contains a space and is not importable by "
                    "its own name; the isolated copy is named SCOPE, which is why this "
                    "check uses one",
            "verification": workspace.verification,
        }
    finally:
        isolation.cleanup_workspace(workspace)


def check_configuration(paths: CampaignPaths, config_path: Path | None = None) -> dict[str, Any]:
    from . import config as config_module
    candidate = Path(config_path or (repo_root() / "conf.yaml"))
    template = repo_root() / "conf.experiment.yaml"
    if not candidate.exists():
        example = repo_root() / "conf.example.yaml"
        parsed = config_module.load_yaml(example) if example.exists() else None
        return {
            "ok": bool(parsed),
            "conf_yaml_present": False,
            "checked_example": str(example) if example.exists() else None,
            "template_present": template.exists(),
            "note": "conf.yaml is intentionally not committed (it holds a key). The "
                    "example parses, so the schema is usable; a formal run needs a "
                    "real conf.yaml produced from conf.experiment.yaml.",
        }
    base = config_module.load_yaml(candidate)
    sample_task = {
        "task_id": "selfcheck-512-r01", "shape_M": 512, "shape_N": 512, "shape_K": 512,
        "variant": "full", "model_role": None,
    }
    effective = config_module.resolve_task_config(base, sample_task, model=None,
                                                base_config_path=candidate)
    return {
        "ok": True,
        "conf_yaml_present": True,
        "config_hash": effective["config_hash"],
        "checks": effective["checks"],
        "matrix_size": effective["derived"]["matrix_size_MNK"],
        "gpu_shapes": effective["derived"]["final_top1_tuning_gpu_shapes"],
    }


def check_inputs() -> dict[str, Any]:
    manifest = data.build_manifest([32], split="development")
    counts = manifest["per_shape_counts"]["32"]
    distinct = len({entry["hashes"]["A"] for entry in manifest["entries"]})
    return {
        "ok": (counts["main"] == 64 and counts["coefficients"] == 128 and distinct >= 32),
        "generator_version": data.GENERATOR_VERSION,
        "entry_count": manifest["entry_count"],
        "per_shape_counts": counts,
        "distinct_A_hashes": distinct,
        "config_count": len(data.CONFIG_IDS),
        "splits": list(data.SPLITS),
        "seed_blocks_disjoint": True,
        "manifest_hash": manifest["manifest_hash"],
    }


def check_reference() -> dict[str, Any]:
    return controls.reference_controls(shape=24)


def check_controls(scratch: Path) -> dict[str, Any]:
    toolchain = build.discover_toolchain("cpu")
    if not toolchain.available:
        return {"ok": False, "reason": "no CPU toolchain to compile the controls",
                "toolchain": toolchain.as_record()}
    cache = reference.ReferenceCache(scratch / "references")
    results = {}
    for control in controls.CPU_CONTROLS:
        if control.name not in ("reference_fp64", "oob_write", "all_zero"):
            continue
        directory = ensure_dir(scratch / control.name)
        results[control.name] = controls.run_control(
            control, directory, shape=32, split="development",
            reference_cache=cache, toolchain=toolchain, include_coefficients=False,
            timeout_seconds=300.0)
    ok = all(record["ok"] for record in results.values())
    return {"ok": ok, "shape": 32,
            "results": {name: {"ok": record["ok"], "observed_main": record.get("observed_main"),
                               "problems": record.get("problems")}
                        for name, record in results.items()}}


def check_parameter_order(scratch: Path) -> dict[str, Any]:
    toolchain = build.discover_toolchain("cpu")
    if not toolchain.available:
        return {"ok": False, "reason": "no CPU toolchain"}
    cache = reference.ReferenceCache(scratch / "references")
    return controls.parameter_order_controls(scratch=scratch / "order", backend="cpu",
                                             toolchain=toolchain,
                                             reference_cache=cache,
                                             timeout_seconds=300.0)


def check_timing(scratch: Path) -> dict[str, Any]:
    toolchain = build.discover_toolchain("cpu")
    if not toolchain.available:
        return {"ok": False, "reason": "no CPU toolchain"}
    kernel_dir = ensure_dir(scratch / "timing_kernel")
    for name in ("cpu_reference_fp64.c",):
        source = controls.CONTROL_SOURCES / name
        (kernel_dir / "cpu_kernel.c").write_bytes(source.read_bytes())
    header = repo_root() / "gemm_code" / "cpu_skeleton" / "kernel.h"
    (kernel_dir / "kernel.h").write_bytes(header.read_bytes())
    build_result = build.build_eval_driver(kernel_dir, scratch / "timing_build",
                                           backend="cpu", toolchain=toolchain)
    if not build_result.ok:
        return {"ok": False, "reason": "driver did not build", "build": build_result.record}
    spec = timing.TimingSpec(label="selfcheck", driver=build_result.binary,
                            build_record=build_result.record, shape=64,
                            backend="cpu", task_id="selfcheck")
    record = timing.time_kernel(spec, scratch / "timing", record_environment=False)
    summary = record["summary"]
    ok = summary["status"] == "measured" and summary["n_sessions"] == 10 and \
        summary["n_launches"] == 500
    return {"ok": ok,
            "sessions": summary["n_sessions"], "launches": summary["n_launches"],
            "gflops_mean": summary["gflops_mean"], "cv": summary["cv"],
            "identity_same_binary": summary["identity"]["same_binary_as_validation"],
            "protocol": record["protocol"]}


def check_toolchains() -> dict[str, Any]:
    cpu = build.discover_toolchain("cpu").as_record()
    cuda = build.discover_toolchain("cuda")
    cuda_record = cuda.as_record()
    if cuda.available:
        cuda_record["arch"] = build.resolve_cuda_arch(cuda.compiler)
    return {"ok": True,
            "cpu": cpu,
            "cuda": cuda_record,
            "sanitizer_available": sanitizer.sanitizer_available(),
            "nvidia_smi": util.which("nvidia-smi"),
            "platform": platform.platform(),
            "gpu_snapshot": util.nvidia_smi_snapshot()}


def run(paths: CampaignPaths, *, config_path: Path | None = None,
        skip_controls: bool = False) -> dict[str, Any]:
    scratch = ensure_dir(paths.workspace_root / "selfcheck")
    checks = [
        _check("package_import", lambda: check_package_import(paths)),
        _check("configuration", lambda: check_configuration(paths, config_path)),
        _check("inputs", check_inputs),
        _check("reference", check_reference),
        _check("toolchains", check_toolchains),
    ]
    if not skip_controls:
        checks.append(_check("controls", lambda: check_controls(scratch)))
        checks.append(_check("parameter_order", lambda: check_parameter_order(scratch)))
        checks.append(_check("timing", lambda: check_timing(scratch)))
    ok = all(check["ok"] for check in checks)
    record = {
        "selfcheck_version": SELFCHECK_VERSION,
        "harness_version": HARNESS_VERSION,
        "created_at": utc_now(),
        "campaign_id": paths.campaign_id,
        "ok": ok,
        "checks": checks,
        "failed": [check["check"] for check in checks if not check["ok"]],
        "hint": ("run `scope-exp controls --backend cuda` for the formal E0 gate; this "
                 "self-check only proves the harness is wired up"
                 if ok else
                 "fix the failed checks before the pilot; do not start a formal queue"),
    }
    ensure_dir(paths.audit)
    write_json(paths.audit / "selfcheck.json", record)
    return record


def render(record: dict[str, Any]) -> str:
    lines = [f"self-check {'PASS' if record['ok'] else 'FAIL'} "
             f"({record['created_at']})"]
    for check in record["checks"]:
        mark = "ok  " if check["ok"] else "FAIL"
        detail = check["details"].get("reason") or ""
        if not detail:
            if check["check"] == "inputs":
                detail = (f"{check['details']['entry_count']} checks, "
                          f"{check['details']['distinct_A_hashes']} distinct A hashes")
            elif check["check"] == "timing":
                detail = (f"{check['details'].get('sessions')} sessions, "
                          f"{check['details'].get('launches')} launches, "
                          f"cv={check['details'].get('cv')}")
            elif check["check"] == "toolchains":
                cpu = check["details"]["cpu"]
                cuda = check["details"]["cuda"]
                detail = (f"cpu={cpu['compiler_kind']} cuda={'yes' if cuda['available'] else 'no'} "
                          f"sanitizer={check['details']['sanitizer_available']}")
            elif check["check"] == "controls":
                detail = str({name: value["ok"] for name, value in
                              check["details"].get("results", {}).items()})
            elif check["check"] == "parameter_order":
                detail = ("declared order passes / legacy order fails"
                          if check["details"].get("ok") else "not detected")
        lines.append(f"  [{mark}] {check['check']}: {detail}")
    if record.get("failed"):
        lines.append(f"failed: {record['failed']}")
    return "\n".join(lines)
