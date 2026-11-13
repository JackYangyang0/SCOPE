from __future__ import annotations
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
from . import data
from .cases import case_file_name, parse_driver_stdout, write_case_file
from .errors import InfraError
from .util import (ensure_dir, nvidia_smi_snapshot, pinned_thread_env, run_command, sha256_file,
                   sha256_value, utc_now, write_csv, write_json)

TIMING_VERSION = "scope-exp-timing-v1"
TIMING_WARMUP_PROCESSES = 2
TIMING_MEASURE_PROCESSES = 10
LAUNCH_WARMUPS = 5
LAUNCH_ITERS = 50
TIMING_CASE = {
    "case_id": "uniform_small.s00",
    "c0_variant": "pattern",
    "coefficients": {"name": "unit", "alpha": 1.0, "beta": 0.0},
    "split": "generation",
}


def gflops(M: int, N: int, K: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return float("nan")
    return 2.0 * M * N * K / (latency_ms * 1.0e6)


def timing_case(shape: int, split: str | None = None) -> dict[str, Any]:
    return data.make_case(TIMING_CASE["case_id"], shape, shape, shape,
                          c0_variant=TIMING_CASE["c0_variant"],
                          split=split or TIMING_CASE["split"],
                          coefficients=TIMING_CASE["coefficients"])


@dataclass
class TimingSpec:
    label: str
    driver: Path
    build_record: dict[str, Any]
    shape: int
    backend: str
    task_id: str = ""
    device: int = 0
    timeout_seconds: float = 600.0
    shape_mnk: tuple[int, int, int] | None = None

    @property
    def binary_sha256(self) -> str | None:
        return self.build_record.get("binary_sha256")

    @property
    def mnk(self) -> tuple[int, int, int]:
        if self.shape_mnk:
            return self.shape_mnk
        return (self.shape, self.shape, self.shape)

    def as_record(self) -> dict[str, Any]:
        M, N, K = self.mnk
        return {
            "label": self.label,
            "task_id": self.task_id,
            "driver": str(self.driver),
            "driver_sha256": sha256_file(self.driver) if Path(self.driver).exists() else None,
            "binary_sha256": self.binary_sha256,
            "backend": self.backend,
            "shape_mnk": [M, N, K],
            "build_argv": self.build_record.get("argv"),
            "kernel_files": self.build_record.get("kernel", {}).get("files"),
            "device": self.device,
        }


@dataclass
class SessionResult:
    spec: TimingSpec
    kind: str
    session_index: int
    raw: list[float] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_latency_ms(self) -> float | None:
        if not self.raw:
            return None
        return float(np.mean(self.raw))

    def summary(self) -> dict[str, Any]:
        M, N, K = self.spec.mnk
        mean = self.mean_latency_ms
        return {
            "task_id": self.spec.task_id,
            "label": self.spec.label,
            "kind": self.kind,
            "session_index": self.session_index,
            "n_launches": len(self.raw),
            "mean_latency_ms": mean,
            "median_latency_ms": float(np.median(self.raw)) if self.raw else None,
            "min_latency_ms": float(np.min(self.raw)) if self.raw else None,
            "max_latency_ms": float(np.max(self.raw)) if self.raw else None,
            "sd_latency_ms": float(np.std(self.raw, ddof=1)) if len(self.raw) > 1 else None,
            "gflops": gflops(M, N, K, mean) if mean else None,
            "status": self.status,
            "error": self.error,
            "binary_sha256": self.spec.binary_sha256,
            "raw_latency_ms": list(self.raw),
            "raw_gflops": [gflops(M, N, K, value) for value in self.raw],
        }

    def rows(self) -> list[dict[str, Any]]:
        M, N, K = self.spec.mnk
        base = {
            "task_id": self.spec.task_id,
            "label": self.spec.label,
            "kind": self.kind,
            "session_index": self.session_index,
            "binary_sha256": self.spec.binary_sha256,
            "shape": f"{M}x{N}x{K}",
            "backend": self.spec.backend,
            "status": self.status,
        }
        return [dict(base, launch_index=index, latency_ms=float(value),
                     gflops=gflops(M, N, K, float(value)))
                for index, value in enumerate(self.raw)]


def run_session(spec: TimingSpec, case_path: Path, *, kind: str,
                session_index: int, cwd: Path | None = None) -> SessionResult:
    result = SessionResult(spec=spec, kind=kind, session_index=session_index)
    if not Path(spec.driver).exists():
        result.status = "missing_binary"
        result.error = f"driver not found: {spec.driver}"
        return result
    argv = [str(spec.driver), "--case", str(case_path), "--mode", "time",
            "--warmup-launches", str(LAUNCH_WARMUPS), "--iters", str(LAUNCH_ITERS)]
    if spec.backend == "cuda":
        argv += ["--device", str(spec.device)]
    else:
        argv += ["--cpu-name", platform.processor() or platform.machine()]
    run = run_command(argv, cwd=cwd, timeout=spec.timeout_seconds,
                      env=pinned_thread_env())
    result.record = {k: run[k] for k in ("argv", "returncode", "timed_out",
                                         "duration_seconds")}
    if run.get("timed_out"):
        result.status = "timeout"
        result.error = "timing session timed out"
        return result
    if run.get("returncode") != 0:
        result.status = "failed"
        result.error = (run.get("stderr") or "")[-400:] or f"exit {run.get('returncode')}"
        return result
    parsed = parse_driver_stdout(run["stdout"])
    if not parsed["found"]:
        result.status = "no_record"
        result.error = parsed.get("error")
        return result
    record = parsed["record"]
    raw = record.get("latencies_ms") or []
    if len(raw) != LAUNCH_ITERS:
        result.status = "short_record"
        result.error = f"expected {LAUNCH_ITERS} launches, got {len(raw)}"
    result.raw = [float(value) for value in raw]
    result.record["driver"] = record
    return result


def time_kernel(spec: TimingSpec, scratch_dir: Path, *, warmup_processes: int = TIMING_WARMUP_PROCESSES,
                measure_processes: int = TIMING_MEASURE_PROCESSES,
                record_environment: bool = True) -> dict[str, Any]:
    ensure_dir(Path(scratch_dir))
    M, N, K = spec.mnk
    case = data.make_case(TIMING_CASE["case_id"], M, N, K,
                          c0_variant=TIMING_CASE["c0_variant"],
                          split=TIMING_CASE["split"],
                          coefficients=TIMING_CASE["coefficients"])
    case_path = Path(scratch_dir) / case_file_name(
        f"timing|{TIMING_CASE['case_id']}|{M}x{N}x{K}")
    write_case_file(case_path, case)

    started_at = utc_now()
    environment_before = nvidia_smi_snapshot() if record_environment else None
    sessions: list[SessionResult] = []
    for index in range(warmup_processes):
        sessions.append(run_session(spec, case_path, kind="warmup",
                                    session_index=index, cwd=scratch_dir))
    for index in range(measure_processes):
        sessions.append(run_session(spec, case_path, kind="measure",
                                    session_index=index, cwd=scratch_dir))
    environment_after = nvidia_smi_snapshot() if record_environment else None

    summary = aggregate_sessions(spec, sessions)
    return {
        "timing_version": TIMING_VERSION,
        "protocol": timing_protocol_record(warmup_processes, measure_processes),
        "spec": spec.as_record(),
        "timing_case": {
            "case_id": TIMING_CASE["case_id"],
            "c0_variant": TIMING_CASE["c0_variant"],
            "coefficients": TIMING_CASE["coefficients"],
            "split": TIMING_CASE["split"],
            "sha256": sha256_file(case_path),
            "selection_rule": "fixed before measurement; not chosen from the test set",
        },
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": (sum(s.record.get("duration_seconds") or 0 for s in sessions)),
        "environment_before": environment_before,
        "environment_after": environment_after,
        "sessions": [session.summary() for session in sessions],
        "summary": summary,
    }


def timing_protocol_record(warmup_processes: int = TIMING_WARMUP_PROCESSES,
                           measure_processes: int = TIMING_MEASURE_PROCESSES) -> dict[str, Any]:
    return {
        "warmup_processes": warmup_processes,
        "measure_processes": measure_processes,
        "launch_warmups": LAUNCH_WARMUPS,
        "launch_iters": LAUNCH_ITERS,
        "timed_region": "kernel call only (one event pair per launch); "
                        "allocations, copies and C re-initialisation are outside",
        "primary_statistic": "untrimmed mean of the session means",
        "also_reported": ["median", "sd", "cv", "n", "raw samples"],
        "trimming_rule": "none: no sample is removed after measurement",
        "throughput_formula": "GFLOP/s = 2*M*N*K/(latency_ms*1e6)",
        "gpu_clock_policy": "recorded per session via nvidia-smi, not pinned by the harness",
    }


def aggregate_sessions(spec: TimingSpec, sessions: Sequence[SessionResult]) -> dict[str, Any]:
    measure = [session for session in sessions
               if session.kind == "measure" and session.status == "ok" and session.raw]
    M, N, K = spec.mnk
    failures = [{"session_index": s.session_index, "status": s.status,
                 "error": s.error} for s in sessions if s.status != "ok"]
    if not measure:
        return {
            "status": "not_measured",
            "n_sessions": 0,
            "n_launches": 0,
            "mean_latency_ms": None,
            "median_latency_ms": None,
            "sd_latency_ms": None,
            "cv": None,
            "gflops_mean": None,
            "gflops_median": None,
            "gflops_sd": None,
            "throughput_metric": "Y_i = delivered * throughput (0 when undelivered)",
            "session_failures": failures,
            "notes": ["no completed measurement session; n=0 must not report a mean"],
        }

    session_means = np.array([s.mean_latency_ms for s in measure], dtype=np.float64)
    session_gflops = np.array([gflops(M, N, K, value) for value in session_means])
    raw = np.array([value for s in measure for value in s.raw], dtype=np.float64)
    sd = float(np.std(session_gflops, ddof=1)) if len(session_gflops) > 1 else None
    return {
        "status": "measured",
        "n_sessions": int(len(measure)),
        "n_launches": int(len(raw)),
        "mean_latency_ms": float(np.mean(session_means)),
        "median_latency_ms": float(np.median(session_means)),
        "sd_latency_ms": float(np.std(session_means, ddof=1)) if len(session_means) > 1 else None,
        "mean_of_raw_latency_ms": float(np.mean(raw)),
        "median_of_raw_latency_ms": float(np.median(raw)),
        "min_latency_ms": float(np.min(raw)),
        "max_latency_ms": float(np.max(raw)),
        "gflops_mean": float(np.mean(session_gflops)),
        "gflops_median": float(np.median(session_gflops)),
        "gflops_sd": sd,
        "gflops_sem": (sd / math.sqrt(len(session_gflops))) if sd is not None else None,
        "cv": (sd / float(np.mean(session_gflops))) if sd is not None else None,
        "session_means_latency_ms": session_means.tolist(),
        "session_gflops": session_gflops.tolist(),
        "session_failures": failures,
        "binary_sha256": spec.binary_sha256,
        "identity": {
            "binary_sha256": spec.binary_sha256,
            "driver_sha256": sha256_file(spec.driver) if Path(spec.driver).exists() else None,
            "build_argv": spec.build_record.get("argv"),
            "kernel_files": spec.build_record.get("kernel", {}).get("files"),
            "same_binary_as_validation": True,
            "note": "validation and timing reuse the same executable, so no "
                    "cross-binary kernel-object proof is needed (§10.2)",
        },
    }


class TimingScheduler:

    def __init__(self, specs: Sequence[TimingSpec], scratch_root: Path,
                 *, warmup_processes: int = TIMING_WARMUP_PROCESSES,
                 measure_processes: int = TIMING_MEASURE_PROCESSES) -> None:
        self.specs = list(specs)
        self.scratch_root = Path(scratch_root)
        self.warmup_processes = warmup_processes
        self.measure_processes = measure_processes
        self.sessions: dict[str, list[SessionResult]] = {spec.label: [] for spec in self.specs}
        self.environment: list[dict[str, Any]] = []

    def case_path(self, spec: TimingSpec) -> Path:
        M, N, K = spec.mnk
        directory = ensure_dir(self.scratch_root / spec.label)
        path = directory / case_file_name(f"timing|{TIMING_CASE['case_id']}|{M}x{N}x{K}")
        if not path.exists():
            write_case_file(path, data.make_case(
                TIMING_CASE["case_id"], M, N, K,
                c0_variant=TIMING_CASE["c0_variant"],
                split=TIMING_CASE["split"],
                coefficients=TIMING_CASE["coefficients"]))
        return path

    def run(self) -> dict[str, dict[str, Any]]:
        rounds = ([("warmup", index) for index in range(self.warmup_processes)]
                  + [("measure", index) for index in range(self.measure_processes)])
        for kind, index in rounds:
            before = nvidia_smi_snapshot()
            if before and before.get("other_compute_processes"):
                self.environment.append({"round": f"{kind}-{index}",
                                         "warning": "another compute process was present",
                                         "snapshot": before})
            for spec in self.specs:
                session = run_session(spec, self.case_path(spec), kind=kind,
                                      session_index=index,
                                      cwd=self.scratch_root / spec.label)
                self.sessions[spec.label].append(session)
            after = nvidia_smi_snapshot()
            if after:
                self.environment.append({"round": f"{kind}-{index}",
                                         "snapshot": after})
        return self.results()

    def results(self) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for spec in self.specs:
            summary = aggregate_sessions(spec, self.sessions[spec.label])
            output[spec.label] = {
                "timing_version": TIMING_VERSION,
                "protocol": timing_protocol_record(self.warmup_processes,
                                                   self.measure_processes),
                "spec": spec.as_record(),
                "sessions": [session.summary() for session in self.sessions[spec.label]],
                "summary": summary,
                "round_environment": self.environment,
                "interleaved_with": [other.label for other in self.specs
                                     if other.label != spec.label],
            }
        return output


def write_timing(output_dir: Path, record: dict[str, Any], *,
                 timing_csv: Path, timing_json: Path) -> dict[str, Path]:
    ensure_dir(Path(output_dir))
    rows: list[dict[str, Any]] = []
    columns = ["task_id", "label", "kind", "session_index", "launch_index",
               "latency_ms", "gflops", "binary_sha256", "shape", "backend"]
    M, N, K = record.get("spec", {}).get("shape_mnk", [None, None, None])
    shape_text = f"{M}x{N}x{K}"
    for session in record.get("sessions", []):
        if session.get("kind") != "measure":
            continue
        raw = session.get("raw_latency_ms") or []
        gflops_values = session.get("raw_gflops") or []
        for index, latency in enumerate(raw):
            rows.append({
                "task_id": session.get("task_id"),
                "label": session.get("label"),
                "kind": session.get("kind"),
                "session_index": session.get("session_index"),
                "launch_index": index,
                "latency_ms": latency,
                "gflops": gflops_values[index] if index < len(gflops_values) else None,
                "binary_sha256": session.get("binary_sha256"),
                "shape": shape_text,
                "backend": record.get("spec", {}).get("backend"),
            })
    return {"timing_csv": write_csv(timing_csv, rows, columns),
            "timing_json": write_json(timing_json, record)}


def collect_rows(sessions: Iterable[SessionResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        rows.extend(session.rows())
    return rows
