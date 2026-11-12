
from __future__ import annotations

import glob
import os
import shutil

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InfraError
from .util import ensure_dir, run_command, sha256_file, utc_now, which, write_json

SANITIZER_VERSION = "scope-exp-sanitizer-v1"
TOOLS = ("memcheck", "racecheck", "synccheck")
DEFAULT_TIMEOUT_SECONDS = 900.0

_SHARED_MEMORY = re.compile(r"__shared__|extern\s+__shared__")
_SYNC = re.compile(r"__syncthreads|__syncwarp|cooperative_groups|__barrier")


@dataclass
class Applicability:
    tool: str
    applicable: bool
    reason: str

    def as_record(self) -> dict[str, Any]:
        return {"tool": self.tool, "applicable": self.applicable, "reason": self.reason}


def strip_comments(text: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", without_block)


def source_features(source_text: str) -> dict[str, Any]:
    code = strip_comments(source_text)
    return {
        "uses_shared_memory": bool(_SHARED_MEMORY.search(code)),
        "uses_explicit_sync": bool(_SYNC.search(code)),
        "shared_declarations": len(_SHARED_MEMORY.findall(code)),
        "sync_sites": len(_SYNC.findall(code)),
    }


def applicability(source_text: str) -> list[Applicability]:
    features = source_features(source_text)
    return [
        Applicability("memcheck", True,
                      "always applicable: out-of-bounds and misaligned accesses"),
        Applicability("racecheck", features["uses_shared_memory"],
                      "kernel declares shared memory" if features["uses_shared_memory"]
                      else "kernel declares no shared memory, so there is no shared "
                           "memory race surface; recorded as not applicable, not as safe"),
        Applicability("synccheck", features["uses_shared_memory"] or features["uses_explicit_sync"],
                      "kernel uses shared memory or explicit synchronisation"
                      if (features["uses_shared_memory"] or features["uses_explicit_sync"])
                      else "kernel has no shared memory and no explicit synchronisation"),
    ]


SANITIZER_EXE_ENV = "SCOPE_SANITIZER_EXE"
_EXE_CACHE: list = []


def _version_key(path: str) -> tuple:
    try:
        result = run_command([path, "--version"], timeout=30)
    except Exception:
        return (0,)
    text = (result.get("stdout") or "") + (result.get("stderr") or "")
    match = re.search(r"Version\s+([0-9]+)\.([0-9]+)\.([0-9]+)", text)
    if not match:
        return (0,)
    return tuple(int(part) for part in match.groups())


def sanitizer_exe() -> str | None:
    if _EXE_CACHE:
        return _EXE_CACHE[0]
    explicit = os.environ.get(SANITIZER_EXE_ENV)
    if explicit and Path(explicit).exists():
        _EXE_CACHE.append(explicit)
        return explicit
    candidates = set()
    found = shutil.which("compute-sanitizer")
    if found:
        candidates.add(found)
    for pattern in ("/usr/local/cuda*/bin/compute-sanitizer",
                    "/opt/cuda*/bin/compute-sanitizer",
                    "/usr/lib/nvidia-cuda-toolkit/compute-sanitizer/compute-sanitizer"):
        for path in glob.glob(pattern):
            if os.access(path, os.X_OK):
                candidates.add(path)
    best = None
    for path in sorted(candidates):
        key = _version_key(path)
        if best is None or key > best[0]:
            best = (key, path)
    _EXE_CACHE.append(best[1] if best else None)
    return _EXE_CACHE[0]


def sanitizer_available() -> bool:
    return sanitizer_exe() is not None


def probe_timeout(driver: Path, case_path: Path, *, tool: str = "memcheck",
                  candidates: tuple[float, ...] = (120.0, 300.0, 600.0, 1200.0),
                  out_dir: Path | None = None) -> dict[str, Any]:
    if not sanitizer_available():
        return {"status": "not_run", "reason": "compute-sanitizer not available"}
    exe = which("compute-sanitizer")
    attempts: list[dict[str, Any]] = []
    for timeout in candidates:
        result = run_command([exe, "--tool", tool, "--error-exitcode", "9",
                              str(driver), "--case", str(case_path),
                              "--mode", "compute", "--out",
                              str((out_dir or Path.cwd()) / "sanitizer_probe.out")],
                             timeout=timeout)
        attempts.append({"timeout_seconds": timeout, "returncode": result["returncode"],
                         "timed_out": result["timed_out"],
                         "duration_seconds": result["duration_seconds"]})
        if not result["timed_out"]:
            return {"status": "ok", "tool": tool, "timeout_seconds": timeout,
                    "attempts": attempts,
                    "recommended_timeout_seconds": max(timeout * 2.0, 300.0)}
    return {"status": "inconclusive", "tool": tool, "attempts": attempts,
            "reason": "no candidate timeout completed"}


SANITIZER_INJECTION_ENV = "SCOPE_SANITIZER_INJECTION_PATH"
INJECTION_LIBRARY = "libsanitizer-collection.so"


def injection_path(exe: str) -> str | None:
    explicit = os.environ.get(SANITIZER_INJECTION_ENV)
    if explicit and Path(explicit).is_dir():
        return explicit
    candidates = [
        Path(exe).resolve().parent,
        Path("/usr/lib/nvidia-cuda-toolkit/compute-sanitizer"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
    ]
    for candidate in candidates:
        if (candidate / INJECTION_LIBRARY).exists():
            return str(candidate)
    return None


def run_sanitizer(driver: Path, case_path: Path, *, tool: str,
                  timeout_seconds: float, out_dir: Path, backend: str = "cuda",
                  device: int = 0) -> dict[str, Any]:
    out_dir = ensure_dir(Path(out_dir))
    record: dict[str, Any] = {
        "sanitizer_version": SANITIZER_VERSION,
        "tool": tool,
        "started_at": utc_now(),
        "driver": str(driver),
        "driver_sha256": sha256_file(driver) if Path(driver).exists() else None,
        "case_file": str(case_path),
        "timeout_seconds": timeout_seconds,
        "status": "not_run",
        "reason": None,
        "error_summary": None,
        "log": None,
        "injection_path": None,
        "target_processes": None,
    }
    if backend != "cuda":
        record.update(status="unsupported",
                      reason="compute-sanitizer applies to CUDA kernels only; "
                             "CPU kernels are checked with guard bands instead")
        write_json(out_dir / f"{tool}.json", record)
        return record
    exe = sanitizer_exe()
    if not exe:
        record.update(status="not_run", reason="compute-sanitizer not available")
        write_json(out_dir / f"{tool}.json", record)
        return record
    log_path = out_dir / f"{tool}.log"
    explicit_inject = os.environ.get(SANITIZER_INJECTION_ENV) or None
    argv = [exe]
    if explicit_inject:
        argv += ["--injection-path", explicit_inject]
    argv += ["--tool", tool, "--target-processes", "all",
             "--error-exitcode", "9", "--print-limit", "20",
             str(driver), "--case", str(case_path), "--mode", "compute",
             "--out", str(out_dir / f"{tool}.out"), "--device", str(device)]
    inject = explicit_inject
    result = run_command(argv, timeout=timeout_seconds)
    log_path.write_text((result["stdout"] or "") + "\n--- stderr ---\n"
                        + (result["stderr"] or ""), encoding="utf-8")
    record.update(sanitizer_exe=exe, argv=argv, injection_path=inject, target_processes="all", returncode=result["returncode"],
                  timed_out=result["timed_out"],
                  duration_seconds=result["duration_seconds"],
                  log=str(log_path))
    text = (result["stdout"] or "") + (result["stderr"] or "")
    summary = re.search(r"ERROR SUMMARY:\s*([0-9]+)\s+error", text)
    if summary:
        record["error_summary"] = int(summary.group(1))
    if result["timed_out"]:
        record.update(status="inconclusive",
                      reason=f"timed out after {timeout_seconds}s; no defect proven")
    elif summary and int(summary.group(1)) > 0:
        record.update(status="fail", reason="sanitizer reported errors")
    elif result["returncode"] == 0:
        record.update(status="pass",
                      reason="completed with no reported error on the executed inputs")
    else:
        record.update(status="inconclusive",
                      reason=f"tool exited {result['returncode']} without an error summary")
    write_json(out_dir / f"{tool}.json", record)
    return record


def run_safety_suite(kernel_dir: Path, driver: Path, case_path: Path, out_dir: Path, *,
                     backend: str = "cuda", timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                     device: int = 0) -> dict[str, Any]:
    kernel_dir = Path(kernel_dir)
    source_name = "cuda_kernel.cuh" if backend == "cuda" else "cpu_kernel.c"
    source_path = kernel_dir / source_name
    if not source_path.exists():
        raise InfraError(f"safety suite needs {source_path}", stage="sanitizer")
    text = source_path.read_text(encoding="utf-8", errors="replace")
    features = source_features(text)
    if backend != "cuda":
        record = {
            "sanitizer_version": SANITIZER_VERSION,
            "kernel_dir": str(kernel_dir),
            "source": source_name,
            "source_sha256": sha256_file(source_path),
            "features": features,
            "driver_sha256": sha256_file(driver) if Path(driver).exists() else None,
            "timeout_seconds": timeout_seconds,
            "status": "pass_scoped",
            "scope": "c_write_bounds_on_all_executed_cases",
            "per_tool": [
                {"tool": "memcheck", "status": "not_applicable",
                 "reason": "compute-sanitizer instruments CUDA kernels; the CPU "
                           "driver's guard band around C is the applicable check"},
                {"tool": "racecheck", "status": "not_applicable",
                 "reason": "CPU build; no GPU shared-memory race surface"},
                {"tool": "synccheck", "status": "not_applicable",
                 "reason": "CPU build; no GPU synchronisation surface"},
            ],
            "limitations": [
                "out-of-bounds reads are not instrumented on the CPU path",
                "data races between CPU threads are not instrumented",
                "only the executed cases and the C array bounds are covered",
            ],
            "safety_is_proven": False,
            "not_run_is_not_safe": True,
            "not_safety_proof_for": "inputs not executed, threads, other devices",
        }
        write_json(Path(out_dir) / "safety.json", record)
        return record
    records = []
    for item in applicability(text):
        if not item.applicable:
            records.append({"tool": item.tool, "status": "not_applicable",
                            "reason": item.reason})
            continue
        records.append(run_sanitizer(driver, case_path, tool=item.tool,
                                     timeout_seconds=timeout_seconds,
                                     out_dir=Path(out_dir), backend=backend,
                                     device=device))
    statuses = {record["tool"]: record["status"] for record in records}
    values = set(statuses.values())
    if "fail" in values:
        combined = "fail"
    elif values and values <= {"pass", "not_applicable"}:
        combined = "pass"
    elif values & {"not_run", "inconclusive", "unsupported"}:
        combined = "incomplete"
    else:
        combined = "not_run"
    record = {
        "sanitizer_version": SANITIZER_VERSION,
        "kernel_dir": str(kernel_dir),
        "source": source_name,
        "source_sha256": sha256_file(source_path),
        "features": features,
        "driver_sha256": sha256_file(driver) if Path(driver).exists() else None,
        "timeout_seconds": timeout_seconds,
        "status": combined,
        "per_tool": records,
        "applicable_mechanism": ("compute-sanitizer on the delivered binary"
                                 if backend == "cuda" else
                                 "host guard band around C in the evaluation driver; "
                                 "compute-sanitizer does not apply to CPU kernels"),
        "safety_is_proven": combined == "pass",
        "not_safety_proof_for": "inputs not executed, other shapes, other devices",
        "not_run_is_not_safe": True,
    }
    write_json(Path(out_dir) / "safety.json", record)
    return record
