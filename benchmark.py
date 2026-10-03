from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "benchmark" / "benchmark_results.json"
EXEEXT = ".exe" if os.name == "nt" else ""
DEFAULT_GPU_PROGRAMS = [
    f"scope={ROOT / '4060ti' / 'results' / 'build' / f'gemm{EXEEXT}'}",
]
DEFAULT_CPU_PROGRAMS = [
    f"scope={ROOT / 'CPU' / 'results' / 'build' / f'gemm_cpu{EXEEXT}'}",
]
DEFAULT_CPU_GENERATION_RESULTS = ROOT / "CPU" / "results"
DEFAULT_GPU_SHAPES = ["512,512,512", "1024,1024,1024", "2048,2048,2048", "4096,4096,4096"]
DEFAULT_CPU_SHAPES = ["256,256,256", "512,512,512", "768,768,768", "1024,1024,1024"]

PRIMARY_PATTERNS = (
    re.compile(
        r"(?:My\s+gemm|SCOPE(?:\s+CPU)?(?:\s+GEMM)?|QiMeng(?:\s+GEMM)?|Custom(?:\s+GEMM)?|Kernel)"
        r"\s+Performance\s*=\s*([0-9.eE+-]+)\s*GFlop/s",
        re.IGNORECASE,
    ),
    re.compile(r"(?:GFLOPS|GFlop/s)\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
    re.compile(r'"gflops"\s*:\s*([0-9.eE+-]+)', re.IGNORECASE),
)
GENERIC_PERFORMANCE_PATTERN = re.compile(
    r"Performance\s*=\s*([0-9.eE+-]+)\s*GFlop/s", re.IGNORECASE
)
LATENCY_PATTERN = re.compile(
    r"(?:My\s+gemm|SCOPE(?:\s+CPU)?(?:\s+GEMM)?|QiMeng(?:\s+GEMM)?|Custom(?:\s+GEMM)?|Kernel)"
    r"\s+Performance\s*=\s*[0-9.eE+-]+\s*GFlop/s,\s*Time\s*=\s*([0-9.eE+-]+)\s*msec",
    re.IGNORECASE,
)
JSON_LATENCY_PATTERN = re.compile(r'"latency_ms"\s*:\s*([0-9.eE+-]+)', re.IGNORECASE)
CUBLAS_PATTERN = re.compile(
    r"(?:cuBLAS|CuBlas|Cublas)\s+Performance\s*=\s*([0-9.eE+-]+)\s*GFlop/s",
    re.IGNORECASE,
)
METRIC_LINE_PATTERN = re.compile(r"(?:SCOPE_METRIC|SCOPE_BASELINE)\s+(.+)", re.IGNORECASE)
ERROR_FIELDS = ("max_abs_error", "max_rel_error", "relative_l2_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compiled GEMM executables with one shared measurement protocol."
    )
    parser.add_argument(
        "--backend",
        choices=("gpu", "cpu"),
        default="cpu",
        help="Benchmark backend. GPU adds reported cuBLAS results; CPU adds an OpenBLAS baseline.",
    )
    parser.add_argument(
        "executables",
        nargs="*",
        help="Executable paths. The filename stem is used as the benchmark name.",
    )
    parser.add_argument(
        "--program",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named executable; repeat to compare scope and multiple baselines.",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        metavar="NAME=TEMPLATE",
        help="Custom command template using {M}, {N}, and {K}; repeat as needed.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="M,N,K",
        help="Problem shape; repeat for multiple shapes. Defaults depend on --backend.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Measured process runs per program and shape.")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Uncounted process runs before measurement.")
    parser.add_argument(
        "--trim",
        type=int,
        default=1,
        help="Remove this many lowest and highest GFLOPS values before aggregation.",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Timeout in seconds for each process run.")
    parser.add_argument("--fp32-atol", type=float, default=1.0e-5, help="Elementwise absolute tolerance reported by the executable (not independently checked here).")
    parser.add_argument("--fp32-rtol", type=float, default=1.0e-3, help="Elementwise relative tolerance reported by the executable (not independently checked here).")
    parser.add_argument(
        "--max-abs-error-tolerance",
        type=float,
        default=1.0e-6,
        help="Require every measured run to report max_abs_error strictly below this value (default: 1e-6).",
    )
    parser.add_argument(
        "--relative-l2-tolerance",
        type=float,
        default=1.0e-3,
        help="Maximum ||C-Cref||_2 / ||Cref||_2 for FP32 GEMM.",
    )
    parser.add_argument(
        "--require-error-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require max_abs_error and relative_l2_error (default: required). Missing metrics never count as a benchmark pass.",
    )
    parser.add_argument(
        "--arg-order",
        choices=("MNK", "MKN", "NMK", "NKM", "KMN", "KNM"),
        default="MKN",
        help="Argument order for --program and positional executables. Default matches scope/QiMeng.",
    )
    parser.add_argument(
        "--metric-regex",
        help="Optional regex whose first capture group is the primary GFLOPS value.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path.")
    parser.add_argument("--csv-output", type=Path, help="CSV summary path; defaults next to the JSON report.")
    parser.add_argument(
        "--no-cublas-baseline",
        action="store_true",
        help="Do not add the cuBLAS measurements reported by the shared benchmark harness as baseline rows.",
    )
    parser.add_argument(
        "--no-openblas-baseline",
        action="store_true",
        help="In CPU mode, do not compile and benchmark the OpenBLAS cblas_sgemm baseline.",
    )
    parser.add_argument(
        "--openblas-root",
        type=Path,
        help="OpenBLAS root containing include/cblas.h and lib or bin; defaults to OPENBLAS_ROOT/config discovery.",
    )
    parser.add_argument(
        "--openblas-baseline-source",
        choices=("generation", "rebenchmark"),
        default="rebenchmark",
        help=(
            "CPU baseline source: generate, compile, and remeasure an OpenBLAS cblas_sgemm "
            "program in this benchmark run (rebenchmark, default), or explicitly reuse the "
            "cpu_blas_gflops recorded during code generation (generation)."
        ),
    )
    parser.add_argument(
        "--cpu-generation-results-root",
        type=Path,
        default=DEFAULT_CPU_GENERATION_RESULTS,
        help="Root containing per-shape verified_ir.json files used by the generation OpenBLAS baseline.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary table.")
    return parser.parse_args()


def split_named_value(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} expects NAME=VALUE, got: {value}")
    name, item = value.split("=", 1)
    if not name.strip() or not item.strip():
        raise ValueError(f"{option} expects non-empty NAME and VALUE: {value}")
    return name.strip(), item.strip()


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in re.split(r"[,xX]", value) if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Shape must be M,N,K, got: {value}")
    shape = tuple(int(part) for part in parts)
    if min(shape) <= 0:
        raise ValueError(f"Shape dimensions must be positive: {value}")
    return shape  # type: ignore[return-value]


def build_programs(args: argparse.Namespace) -> list[dict[str, Any]]:
    programs = []
    for raw_path in args.executables:
        path = Path(raw_path).expanduser().resolve()
        programs.append({"name": path.stem, "path": path, "template": None})
    default_programs = DEFAULT_CPU_PROGRAMS if args.backend == "cpu" else DEFAULT_GPU_PROGRAMS
    configured_programs = args.program or ([] if args.executables or args.command else default_programs)
    for value in configured_programs:
        name, raw_path = split_named_value(value, "--program")
        programs.append({"name": name, "path": Path(raw_path).expanduser().resolve(), "template": None})
    for value in args.command:
        name, template = split_named_value(value, "--command")
        programs.append({"name": name, "path": None, "template": template})
    if not programs:
        raise ValueError("Provide at least one executable or --program/--command entry.")
    names = [program["name"] for program in programs]
    if len(names) != len(set(names)):
        raise ValueError("Program names must be unique.")
    for program in programs:
        path = program.get("path")
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Executable not found: {path}")
    return programs


def prepare_openblas_program(args: argparse.Namespace) -> dict[str, Any]:
    from verification.c_build_run_verifier import (
        DEFAULT_OPENBLAS_BASELINE_SOURCE,
        compile_openblas_baseline,
        resolve_openblas_root,
    )

    openblas_root = resolve_openblas_root(args.openblas_root)
    if openblas_root is None:
        raise ValueError(
            "OpenBLAS root not found. Pass --openblas-root or set OPENBLAS_ROOT; "
            "use --no-openblas-baseline to disable it."
        )
    include_dir = openblas_root / "include"
    lib_dir = openblas_root / "lib"
    bin_dir = openblas_root / "bin"
    build_dir = args.output.expanduser().resolve().parent / "openblas"
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "openblas_sgemm_baseline.c"
    executable = build_dir / ("openblas_sgemm_baseline.exe" if os.name == "nt" else "openblas_sgemm_baseline")
    shutil.copyfile(DEFAULT_OPENBLAS_BASELINE_SOURCE, source_path)
    compile_result = compile_openblas_baseline(
        source_path,
        executable,
        include_dir,
        lib_dir,
        int(args.timeout),
        "windows" if os.name == "nt" else "linux",
    )
    if compile_result.get("status") != "pass":
        detail = compile_result.get("stderr") or compile_result.get("stdout") or "unknown compiler error"
        raise ValueError(f"Failed to build OpenBLAS baseline: {detail}")
    env = {}
    if bin_dir.is_dir():
        env["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    thread_text = str(max(1, os.cpu_count() or 1))
    env["OPENBLAS_NUM_THREADS"] = thread_text
    env["GOTO_NUM_THREADS"] = thread_text
    env["OMP_NUM_THREADS"] = thread_text
    env["OMP_DYNAMIC"] = "FALSE"
    return {
        "name": "OpenBLAS",
        "path": executable,
        "template": None,
        "baseline_kind": "openblas_cblas_sgemm",
        "arg_order": "MKN",
        "env": env,
        "thread_count": int(thread_text),
    }


def generated_openblas_baselines(
    shapes: list[tuple[int, int, int]],
    results_root: Path,
) -> list[dict[str, Any]]:
    baselines = []
    results_root = results_root.expanduser().resolve()
    for shape in shapes:
        m, n, k = shape
        shape_name = str(m) if m == n == k else f"{m}x{n}x{k}"
        ir_path = results_root / shape_name / "verified_ir.json"
        if not ir_path.is_file():
            raise FileNotFoundError(
                f"Generated CPU baseline record not found: {ir_path}. "
                "Generate this shape first, pass --cpu-generation-results-root, "
                "or use --openblas-baseline-source rebenchmark."
            )
        ir = json.loads(ir_path.read_text(encoding="utf-8"))
        performance = ir.get("performance", {}) or {}
        verification = ir.get("verification", {}) or {}
        optimized = verification.get("optimized_cpu_baseline", {}) or {}
        gflops = performance.get("cpu_blas_gflops")
        latency_ms = performance.get("cpu_blas_latency_ms", optimized.get("latency_ms"))
        if not isinstance(gflops, (int, float)) or not math.isfinite(gflops) or gflops <= 0:
            raise ValueError(f"No valid performance.cpu_blas_gflops in {ir_path}")
        command = optimized.get("command") or []
        baselines.append({
            "program": "OpenBLAS",
            "baseline_kind": "recorded_during_cpu_generation",
            "baseline_source": "generation_verified_ir",
            "source_ir": str(ir_path),
            "executable": str(command[0]) if command else None,
            "command": command,
            "shape": {"M": m, "N": n, "K": k},
            "status": "pass",
            "correctness": "reference",
            "error_validation": {"status": "reference"},
            "warmup_runs": 0,
            "requested_runs": 0,
            "successful_runs": 1,
            "gflops": summarize_values([float(gflops)], 0),
            "latency_ms": (
                summarize_values([float(latency_ms)], 0)
                if isinstance(latency_ms, (int, float)) and math.isfinite(latency_ms)
                else None
            ),
            "reported_cublas_gflops": None,
            "runs": [],
            "warmups": [],
        })
    return baselines


def command_for(program: dict[str, Any], shape: tuple[int, int, int], arg_order: str) -> list[str]:
    m, n, k = shape
    values = {"M": m, "N": n, "K": k}
    template = program.get("template")
    if template:
        rendered = template.format(**values)
        return shlex.split(rendered, posix=os.name != "nt")
    path = program["path"]
    effective_arg_order = program.get("arg_order", arg_order)
    return [str(path), *(str(values[key]) for key in effective_arg_order)]


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_metrics(text: str, custom_pattern: re.Pattern[str] | None = None) -> dict[str, Any]:
    gflops = None
    patterns = (custom_pattern,) if custom_pattern is not None else PRIMARY_PATTERNS
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            gflops = finite_float(match.group(1))
            if gflops is not None:
                break
    if gflops is None:
        for line in text.splitlines():
            if "cublas" in line.lower():
                continue
            match = GENERIC_PERFORMANCE_PATTERN.search(line)
            if match:
                gflops = finite_float(match.group(1))
                if gflops is not None:
                    break

    latency_match = LATENCY_PATTERN.search(text) or JSON_LATENCY_PATTERN.search(text)
    cublas_match = CUBLAS_PATTERN.search(text)
    correctness = "unknown"
    if re.search(r"Result\s*=\s*PASS|\bCORRECT(?:NESS)?\s*[:=]\s*PASS\b", text, re.IGNORECASE):
        correctness = "pass"
    if re.search(r"Result\s*=\s*FAIL|\bCORRECT(?:NESS)?\s*[:=]\s*FAIL\b", text, re.IGNORECASE):
        correctness = "fail"
    parsed = {
        "gflops": gflops,
        "latency_ms": finite_float(latency_match.group(1)) if latency_match else None,
        "cublas_gflops": finite_float(cublas_match.group(1)) if cublas_match else None,
        "correctness": correctness,
    }
    metric_line = METRIC_LINE_PATTERN.search(text)
    if metric_line:
        for key, raw_value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", metric_line.group(1)):
            value = finite_float(raw_value)
            if value is not None:
                parsed[key] = value
            elif key == "correctness":
                parsed[key] = raw_value.lower()
    for field in ERROR_FIELDS:
        if field in parsed:
            continue
        match = re.search(rf"\b{field}\s*[:=]\s*([0-9.eE+-]+)", text, re.IGNORECASE)
        if match:
            parsed[field] = finite_float(match.group(1))
    return parsed


def validate_error_metrics(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    available = [
        field for field in ERROR_FIELDS
        if isinstance(metrics.get(field), (int, float)) and math.isfinite(metrics[field])
    ]
    required = {"max_abs_error", "relative_l2_error"}
    missing = sorted(required - set(available))
    failures = []
    max_abs_error = metrics.get("max_abs_error")
    if "max_abs_error" in available and (max_abs_error < 0 or max_abs_error >= args.max_abs_error_tolerance):
        failures.append(
            f"max_abs_error {max_abs_error:.6e} is not below {args.max_abs_error_tolerance:.6e}"
        )
    relative_l2 = metrics.get("relative_l2_error")
    if "relative_l2_error" in available and (relative_l2 < 0 or relative_l2 > args.relative_l2_tolerance):
        failures.append(
            f"relative_l2_error {relative_l2:.6e} exceeds {args.relative_l2_tolerance:.6e}"
        )
    if metrics.get("correctness") == "fail":
        failures.append("program reported correctness failure")
    if args.require_error_metrics and missing:
        failures.append("missing required error metrics: " + ", ".join(missing))
    return {
        "status": "fail" if failures else ("pass" if not missing else "unverified"),
        "reference": "program_reported_reference",
        "fp32_atol": args.fp32_atol,
        "fp32_rtol": args.fp32_rtol,
        "elementwise_tolerances_checked_by_benchmark": False,
        "max_abs_error_tolerance": args.max_abs_error_tolerance,
        "relative_l2_tolerance": args.relative_l2_tolerance,
        "available_metrics": available,
        "missing_metrics": missing,
        "failures": failures,
    }


def run_once(
    command: list[str],
    cwd: Path,
    timeout: float,
    metric_pattern: re.Pattern[str] | None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        env = os.environ.copy()
        env.update(env_overrides or {})
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        metrics = parse_metrics(output, metric_pattern)
        return {
            "status": "pass" if completed.returncode == 0 and metrics["gflops"] is not None else "fail",
            "returncode": completed.returncode,
            "metrics": metrics,
            "output": output,
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        return {
            "status": "timeout",
            "returncode": None,
            "metrics": {},
            "output": "\n".join(part for part in (stdout, stderr) if part),
        }
    except OSError as error:
        return {"status": "error", "returncode": None, "metrics": {}, "output": str(error)}


def summarize_values(values: list[float], trim: int) -> dict[str, Any]:
    ordered = sorted(values)
    used = ordered[trim : len(ordered) - trim] if trim else ordered
    if not used:
        return {"count": 0, "raw_count": len(values), "trim_each_side": trim}
    return {
        "count": len(used),
        "raw_count": len(values),
        "trim_each_side": trim,
        "mean": statistics.fmean(used),
        "median": statistics.median(used),
        "stddev": statistics.pstdev(used),
        "min": min(used),
        "max": max(used),
        "values": used,
        "raw_values": values,
    }


def benchmark_case(
    program: dict[str, Any],
    shape: tuple[int, int, int],
    args: argparse.Namespace,
    metric_pattern: re.Pattern[str] | None,
) -> dict[str, Any]:
    command = command_for(program, shape, args.arg_order)
    cwd = program["path"].parent if program.get("path") else ROOT
    is_reference_baseline = bool(program.get("baseline_kind"))
    effective_metric_pattern = None if is_reference_baseline else metric_pattern
    warmups = []
    measured = []
    for index in range(args.warmup_runs):
        result = run_once(command, cwd, args.timeout, effective_metric_pattern, program.get("env"))
        warmups.append(result)
        if not args.quiet:
            print(f"[{program['name']} {shape}] warmup {index + 1}/{args.warmup_runs}: {result['status']}", flush=True)
        if result["status"] != "pass":
            break
    if all(result["status"] == "pass" for result in warmups):
        for index in range(args.runs):
            result = run_once(command, cwd, args.timeout, effective_metric_pattern, program.get("env"))
            measured.append(result)
            value = result.get("metrics", {}).get("gflops")
            if not args.quiet:
                shown = f"{value:.3f} GFLOPS" if isinstance(value, (int, float)) else result["status"]
                print(f"[{program['name']} {shape}] run {index + 1}/{args.runs}: {shown}", flush=True)

    successful = [result for result in measured if result["status"] == "pass"]
    gflops_values = [result["metrics"]["gflops"] for result in successful]
    latency_values = [
        result["metrics"]["latency_ms"]
        for result in successful
        if result["metrics"].get("latency_ms") is not None
    ]
    cublas_values = [
        result["metrics"]["cublas_gflops"]
        for result in successful
        if result["metrics"].get("cublas_gflops") is not None
    ]
    correctness_values = [result["metrics"].get("correctness") for result in successful]
    correctness = "fail" if "fail" in correctness_values else (
        "pass" if correctness_values and all(value == "pass" for value in correctness_values) else "unknown"
    )
    if is_reference_baseline:
        correctness = "reference"
    error_validations = [validate_error_metrics(result["metrics"], args) for result in successful]
    error_status = "fail" if any(item["status"] == "fail" for item in error_validations) else (
        "pass" if error_validations and all(item["status"] == "pass" for item in error_validations) else "unverified"
    )
    if is_reference_baseline:
        error_status = "reference"
    error_statistics = {
        field: summarize_values(
            [result["metrics"][field] for result in successful if isinstance(result["metrics"].get(field), (int, float))],
            0,
        )
        for field in ERROR_FIELDS
    }
    return {
        "program": program["name"],
        "baseline_kind": program.get("baseline_kind"),
        "executable": str(program["path"]) if program.get("path") else None,
        "command": command,
        "shape": {"M": shape[0], "N": shape[1], "K": shape[2]},
        "status": "pass" if len(successful) == args.runs and (
            is_reference_baseline or (correctness == "pass" and error_status == "pass")
        ) else "fail",
        "correctness": correctness,
        "error_validation": {
            "status": error_status,
            "fp32_atol": args.fp32_atol,
            "fp32_rtol": args.fp32_rtol,
            "elementwise_tolerances_checked_by_benchmark": False,
            "max_abs_error_tolerance": args.max_abs_error_tolerance,
            "relative_l2_tolerance": args.relative_l2_tolerance,
            "require_error_metrics": args.require_error_metrics,
            "statistics": error_statistics,
            "runs": error_validations,
        },
        "warmup_runs": len(warmups),
        "requested_runs": args.runs,
        "successful_runs": len(successful),
        "gflops": summarize_values(gflops_values, args.trim),
        "latency_ms": summarize_values(latency_values, args.trim) if latency_values else None,
        "reported_cublas_gflops": summarize_values(cublas_values, args.trim) if cublas_values else None,
        "runs": measured,
        "warmups": warmups,
    }


def reported_cublas_baselines(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = []
    seen_shapes: set[tuple[int, int, int]] = set()
    if any(str(result.get("program", "")).lower() == "cublas" for result in results):
        return baselines
    for result in results:
        shape = result["shape"]
        shape_key = (shape["M"], shape["N"], shape["K"])
        cublas = result.get("reported_cublas_gflops") or {}
        if shape_key in seen_shapes or result.get("status") != "pass" or not cublas.get("count"):
            continue
        seen_shapes.add(shape_key)
        baselines.append({
            "program": "cuBLAS",
            "baseline_kind": "reported_by_shared_harness",
            "source_program": result["program"],
            "executable": None,
            "command": result.get("command"),
            "shape": dict(shape),
            "status": "pass",
            "correctness": "reference",
            "warmup_runs": result.get("warmup_runs", 0),
            "requested_runs": cublas.get("raw_count", cublas.get("count", 0)),
            "successful_runs": cublas.get("raw_count", cublas.get("count", 0)),
            "gflops": cublas,
            "latency_ms": None,
            "reported_cublas_gflops": None,
            "runs": [],
            "warmups": [],
        })
    return baselines


def add_relative_to_openblas(results: list[dict[str, Any]]) -> None:
    baseline_by_shape = {}
    for result in results:
        if str(result.get("program", "")).lower() != "openblas":
            continue
        shape = result["shape"]
        mean = (result.get("gflops") or {}).get("mean")
        if isinstance(mean, (int, float)) and mean > 0:
            baseline_by_shape[(shape["M"], shape["N"], shape["K"])] = mean
    for result in results:
        shape = result["shape"]
        baseline = baseline_by_shape.get((shape["M"], shape["N"], shape["K"]))
        mean = (result.get("gflops") or {}).get("mean")
        result["relative_to_openblas"] = (
            mean / baseline
            if result.get("program") != "OpenBLAS"
            and isinstance(mean, (int, float))
            and isinstance(baseline, (int, float))
            and baseline > 0
            else None
        )


def write_reports(report: dict[str, Any], output: Path, csv_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "program", "M", "N", "K", "status", "correctness", "successful_runs",
                "gflops_mean", "gflops_median", "gflops_stddev", "gflops_min", "gflops_max",
                "latency_ms_mean", "reported_cublas_gflops_mean",
                "relative_to_openblas",
                "max_abs_error", "max_rel_error", "relative_l2_error", "error_validation",
            ),
        )
        writer.writeheader()
        for result in report["results"]:
            stats = result.get("gflops", {})
            latency = result.get("latency_ms") or {}
            cublas = result.get("reported_cublas_gflops") or {}
            shape = result["shape"]
            error = result.get("error_validation") or {}
            error_stats = error.get("statistics") or {}
            writer.writerow({
                "program": result["program"], "M": shape["M"], "N": shape["N"], "K": shape["K"],
                "status": result["status"], "correctness": result["correctness"],
                "successful_runs": result["successful_runs"], "gflops_mean": stats.get("mean"),
                "gflops_median": stats.get("median"), "gflops_stddev": stats.get("stddev"),
                "gflops_min": stats.get("min"), "gflops_max": stats.get("max"),
                "latency_ms_mean": latency.get("mean"),
                "reported_cublas_gflops_mean": cublas.get("mean"),
                "relative_to_openblas": result.get("relative_to_openblas"),
                "max_abs_error": (error_stats.get("max_abs_error") or {}).get("max"),
                "max_rel_error": (error_stats.get("max_rel_error") or {}).get("max"),
                "relative_l2_error": (error_stats.get("relative_l2_error") or {}).get("max"),
                "error_validation": error.get("status"),
            })


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nProgram                  Shape              Runs   Mean GFLOPS   Median   Stddev   vs OpenBLAS   Status    Correct    MaxAbs     RelL2")
    print("-" * 152)
    for result in results:
        shape = result["shape"]
        shape_text = f"{shape['M']}x{shape['N']}x{shape['K']}"
        stats = result.get("gflops", {})
        relative_l2 = (((result.get("error_validation") or {}).get("statistics") or {}).get("relative_l2_error") or {}).get("max")
        max_abs_error = (((result.get("error_validation") or {}).get("statistics") or {}).get("max_abs_error") or {}).get("max")
        relative_l2_text = f"{relative_l2:.3e}" if isinstance(relative_l2, (int, float)) else "unverified"
        max_abs_text = f"{max_abs_error:.3e}" if isinstance(max_abs_error, (int, float)) else "unverified"
        relative_to_openblas = result.get("relative_to_openblas")
        openblas_text = f"{relative_to_openblas:.3f}x" if isinstance(relative_to_openblas, (int, float)) else "-"
        print(
            f"{result['program'][:24]:24} {shape_text:18} "
            f"{result['successful_runs']:>2}/{result['requested_runs']:<2} "
            f"{stats.get('mean', float('nan')):>13.3f} "
            f"{stats.get('median', float('nan')):>9.3f} "
            f"{stats.get('stddev', float('nan')):>8.3f} "
            f"{openblas_text:>13}   {result['status']:<8} {result['correctness']:<8} {max_abs_text:>10} {relative_l2_text}"
        )


def main() -> int:
    args = parse_args()
    if args.runs < 1 or args.warmup_runs < 0 or args.trim < 0 or args.timeout <= 0:
        raise ValueError("runs/timeout must be positive; warmup-runs/trim must be non-negative.")
    if args.trim * 2 >= args.runs:
        raise ValueError("--trim removes all measured runs; require 2 * trim < runs.")
    if min(args.fp32_atol, args.fp32_rtol, args.relative_l2_tolerance) < 0 or args.max_abs_error_tolerance <= 0:
        raise ValueError("FP32 error tolerances must be non-negative.")
    default_shapes = DEFAULT_CPU_SHAPES if args.backend == "cpu" else DEFAULT_GPU_SHAPES
    shapes = [parse_shape(value) for value in (args.shape or default_shapes)]
    programs = build_programs(args)
    if (
        args.backend == "cpu"
        and not args.no_openblas_baseline
        and args.openblas_baseline_source == "rebenchmark"
    ):
        if any(program["name"].lower() == "openblas" for program in programs):
            raise ValueError("Program name OpenBLAS is reserved for the automatic CPU baseline.")
        programs.append(prepare_openblas_program(args))
    metric_pattern = re.compile(args.metric_regex, re.IGNORECASE) if args.metric_regex else None
    results = [
        benchmark_case(program, shape, args, metric_pattern)
        for shape in shapes
        for program in programs
    ]
    if args.backend == "gpu" and not args.no_cublas_baseline:
        results.extend(reported_cublas_baselines(results))
    if args.backend == "cpu" and not args.no_openblas_baseline:
        if args.openblas_baseline_source == "generation":
            results.extend(generated_openblas_baselines(shapes, args.cpu_generation_results_root))
        add_relative_to_openblas(results)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "backend": args.backend,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "trim_each_side": args.trim,
            "timeout_seconds": args.timeout,
            "argument_order": args.arg_order,
            "include_reported_cublas_baseline": args.backend == "gpu" and not args.no_cublas_baseline,
            "include_openblas_baseline": args.backend == "cpu" and not args.no_openblas_baseline,
            "openblas_baseline_source": (
                args.openblas_baseline_source
                if args.backend == "cpu" and not args.no_openblas_baseline
                else None
            ),
            "error_policy": {
                "dtype": "fp32",
                "elementwise_atol": args.fp32_atol,
                "elementwise_rtol": args.fp32_rtol,
                "elementwise_tolerances_checked_by_benchmark": False,
                "max_abs_error_tolerance": args.max_abs_error_tolerance,
                "max_abs_error_rule": "strictly_less_than",
                "relative_l2_tolerance": args.relative_l2_tolerance,
                "require_error_metrics": args.require_error_metrics,
            },
        },
        "results": results,
    }
    output = args.output.resolve()
    csv_output = (args.csv_output or output.with_suffix(".csv")).resolve()
    write_reports(report, output, csv_output)
    print_summary(results)
    print(f"\nJSON: {output}")
    print(f"CSV:  {csv_output}")
    return 0 if all(result["status"] == "pass" for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, re.error) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(2)
