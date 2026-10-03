"""Tune an existing SCOPE GEMM implementation; CUDA is the default backend."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from SCOPE.generate_ir.ir_extraction import build_extracted_ir
from SCOPE.specialization.shape_search import passed, write_json
from SCOPE.tune_cuda import aggregate_value, clear_results
from SCOPE.tune_cuda import tune_sources as tune_cuda_sources
from SCOPE.utils.common_utils import load_config
from SCOPE.verification.c_build_run_verifier import run_cpu_optimized_baseline, verify_cpu_build_and_run

ROOT = Path(__file__).resolve().parent
CPU_TILE_KEYS = ("L2_M", "L2_N", "L2_K", "L1_M", "L1_N", "L1_K", "MR", "NR")
CPU_SYMBOLS = {
    "L2_M": ("L2_BLOCK_M", "MC"),
    "L2_N": ("L2_BLOCK_N", "NC"),
    "L2_K": ("L2_BLOCK_K", "KC"),
    "L1_M": ("L1_BLOCK_M", "M1"),
    "L1_N": ("L1_BLOCK_N", "N1"),
    "L1_K": ("L1_BLOCK_K", "K1"),
    "MR": ("MR",),
    "NR": ("NR",),
}


def read_sources(source: Path, backend: str = "cuda") -> tuple[Path, str, dict[str, str]]:
    source = Path(source).resolve()
    if not source.exists():
        raise ValueError(f"Source does not exist: {source}")
    folder = source if source.is_dir() else source.parent
    if backend == "cpu":
        kernel = "cpu_kernel.c" if source.is_dir() else source.name
        harness = "main.c"
        suffixes = {".c", ".h"}
    else:
        kernel = "cuda_kernel.cuh" if source.is_dir() else source.name
        harness = "main.cpp"
        suffixes = {".cu", ".cuh", ".h", ".hpp", ".cpp"}
    sources = {
        path.name: path.read_text(encoding="utf-8-sig")
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    if kernel not in sources or harness not in sources:
        raise ValueError(f"{backend.upper()} source must contain {kernel} and {harness}")
    return folder, kernel, sources


def cpu_strategy_domains(library: dict[str, Any]) -> dict[str, list[tuple[int, ...]]]:
    domains: dict[str, list[tuple[int, ...]]] = {"l2": [], "l1": [], "register": []}
    for stage in library.get("stages", []) or []:
        for subphase in stage.get("ordered_subphases", []) or []:
            for strategy in subphase.get("strategies", []) or []:
                updates = strategy.get("ir_updates", {}) or {}
                if all(key in updates for key in (
                    "cpu_tiling.l2_block_m", "cpu_tiling.l2_block_n", "cpu_tiling.l2_block_k"
                )):
                    domains["l2"].append(tuple(int(updates[key]) for key in (
                        "cpu_tiling.l2_block_m", "cpu_tiling.l2_block_n", "cpu_tiling.l2_block_k"
                    )))
                if all(key in updates for key in (
                    "cpu_tiling.l1_block_m", "cpu_tiling.l1_block_n", "cpu_tiling.l1_block_k"
                )):
                    domains["l1"].append(tuple(int(updates[key]) for key in (
                        "cpu_tiling.l1_block_m", "cpu_tiling.l1_block_n", "cpu_tiling.l1_block_k"
                    )))
                if all(key in updates for key in ("cpu_tiling.register_m", "cpu_tiling.register_n")):
                    domains["register"].append(tuple(int(updates[key]) for key in (
                        "cpu_tiling.register_m", "cpu_tiling.register_n"
                    )))
    return {name: list(dict.fromkeys(values)) for name, values in domains.items()}


def extract_cpu_tile(source: str) -> dict[str, int]:
    tile: dict[str, int] = {}
    for key, aliases in CPU_SYMBOLS.items():
        for symbol in aliases:
            match = re.search(rf"\b{re.escape(symbol)}\s*=\s*(\d+)\b", source)
            if match:
                tile[key] = int(match.group(1))
                break
    return tile


def cpu_axis_values(seed_values: list[int], shape_extent: int, standard: tuple[int, ...]) -> list[int]:
    values = {int(value) for value in seed_values if int(value) > 0}
    values.update(value for value in standard if value <= shape_extent)
    # Shape divisors avoid tail work; standard non-divisors remain available
    # because cache/parallelism can outweigh a small cleanup tile.
    values.update(value for value in range(16, min(shape_extent, 512) + 1, 16) if shape_extent % value == 0)
    return sorted(values)


def build_cpu_tile_space(
    library: dict[str, Any],
    source: str,
    ir: dict[str, Any] | None = None,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    current = extract_cpu_tile(source)
    domains = cpu_strategy_domains(library)
    problem = (ir or {}).get("problem", {}) or {}
    m = max(1, int(problem.get("M") or 512))
    n = max(1, int(problem.get("N") or 512))
    k = max(1, int(problem.get("K") or 512))
    l2_m = cpu_axis_values([value[0] for value in domains["l2"]], m, (32, 64, 96, 128, 192, 256, 384, 512))
    l2_n = cpu_axis_values([value[1] for value in domains["l2"]], n, (32, 64, 96, 128, 192, 256, 384, 512))
    l2_k = cpu_axis_values([value[2] for value in domains["l2"]], k, (16, 32, 48, 64, 96, 128, 192, 256))
    l2_values = list(product(l2_m, l2_n, l2_k))

    has_l1_materialization = all(key in current for key in ("L1_M", "L1_N", "L1_K"))
    if has_l1_materialization:
        l1_m = cpu_axis_values([value[0] for value in domains["l1"]], m, (8, 16, 24, 32, 48, 64))
        l1_n = cpu_axis_values([value[1] for value in domains["l1"]], n, (8, 16, 24, 32, 48, 64))
        l1_k = cpu_axis_values([value[2] for value in domains["l1"]], k, (16, 32, 48, 64, 96, 128))
        l1_values = list(product(l1_m, l1_n, l1_k))
    else:
        l1_values = [tuple(current.get(key, value) for key, value in zip(("L1_M", "L1_N", "L1_K"), (32, 32, 64)))]
    # Explicit SIMD micro-kernels encode the register tile in their accumulator
    # declarations and FMA/store statements. Replacing MR/NR alone would only
    # change loop steps, not the unrolled body. Such kernels may opt in after a
    # structural register-tile materializer is provided.
    register_tunable = "SCOPE_CPU_TUNABLE_REGISTER_TILE" in source
    register_values = (
        domains["register"]
        if register_tunable and domains["register"]
        else [(current.get("MR", 4), current.get("NR", 8))]
    )
    hardware = (ir or {}).get("hardware", {}) or {}
    cache = hardware.get("cpu_cache", {}) or {}
    l1_budget = int(cache.get("l1_data_cache_bytes") or 32 * 1024) * 3 // 4
    l2_budget = int(cache.get("l2_cache_bytes") or 1024 * 1024) * 4 // 5
    cores = max(1, int(hardware.get("cpu_physical_cores") or 1))
    candidates = []
    rejected_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    for l2, l1, register in product(l2_values, l1_values, register_values):
        tile = dict(zip(CPU_TILE_KEYS, (*l2, *l1, *register)))
        if any(tile[key] <= 0 for key in CPU_TILE_KEYS):
            reject("non_positive")
            continue
        if any(tile[l1_key] > tile[l2_key] for l1_key, l2_key in (
            ("L1_M", "L2_M"), ("L1_N", "L2_N"), ("L1_K", "L2_K")
        )):
            reject("l1_exceeds_l2")
            continue
        if tile["MR"] > tile["L1_M"] or tile["NR"] > tile["L1_N"]:
            reject("register_exceeds_l1")
            continue
        l1_bytes = 4 * (tile["L1_M"] * tile["L1_K"] + tile["L1_K"] * tile["L1_N"])
        l2_bytes = 4 * (
            tile["L2_M"] * tile["L2_K"]
            + tile["L2_K"] * tile["L2_N"]
            + tile["L2_M"] * tile["L2_N"]
        )
        output_tasks = ((m + tile["L2_M"] - 1) // tile["L2_M"]) * ((n + tile["L2_N"] - 1) // tile["L2_N"])
        tile["estimated_l1_bytes"] = l1_bytes
        tile["estimated_l2_bytes"] = l2_bytes
        tile["estimated_output_tasks"] = output_tasks
        tile["l1_cache_pressure"] = l1_bytes > l1_budget
        tile["l2_cache_pressure"] = l2_bytes > l2_budget
        tile["parallelism_pressure"] = output_tasks < cores
        candidates.append(tile)
    audit = {
        "axis_values": {
            "L2_M": l2_m, "L2_N": l2_n, "L2_K": l2_k,
            "L1": l1_values,
            "register": register_values,
        },
        "cartesian_candidate_count": len(l2_values) * len(l1_values) * len(register_values),
        "accepted_candidate_count": len(candidates),
        "rejected_counts": rejected_counts,
        "cache_budgets": {"l1_bytes": l1_budget, "l2_bytes": l2_budget},
        "physical_cores": cores,
        "l1_materialized_by_source": has_l1_materialization,
        "register_tile_materialized_by_source": register_tunable,
    }
    return candidates, audit


def enumerate_cpu_tiles(
    library: dict[str, Any],
    source: str,
    ir: dict[str, Any] | None = None,
) -> list[dict[str, int]]:
    return build_cpu_tile_space(library, source, ir)[0]


def specialize_cpu_source(source: str, tile: dict[str, int]) -> str | None:
    updated = source
    changed = False
    for key, value in tile.items():
        if key not in CPU_SYMBOLS:
            continue
        aliases = CPU_SYMBOLS[key]
        for symbol in aliases:
            pattern = rf"(\b(?:static\s+)?const\s+int\s+{re.escape(symbol)}\s*=\s*)\d+\b"
            updated, count = re.subn(pattern, rf"\g<1>{int(value)}", updated, count=1)
            if count:
                changed = True
                break
    return updated if changed else None


def tune_cpu_sources(
    sources: dict[str, str],
    kernel: str,
    ir: dict[str, Any],
    library: dict[str, Any],
    output: Path,
    *,
    max_candidates: int = 0,
    rounds: int = 3,
    benchmark_runs: int = 5,
    warmup_runs: int = 2,
    platform: str = "windows",
    timeout: int = 180,
    target_ratio: float | None = None,
    aggregation: str = "mean",
    openblas_root: Path | None = None,
    verify: Callable[..., dict[str, Any]] = verify_cpu_build_and_run,
    candidate_tiles: list[dict[str, int]] | None = None,
    screening_benchmark_runs: int = 1,
    screening_warmup_runs: int = 0,
    final_remeasure_top_k: int = 3,
) -> dict[str, Any]:
    tuning_started = time.perf_counter()
    if max_candidates < 0 or rounds < 1:
        raise ValueError("Candidate budget must be nonnegative; rounds must be positive")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty to avoid mixing old measurements: {output}")
    output.mkdir(parents=True, exist_ok=True)
    baseline = clear_results(ir)
    baseline.setdefault("target", {}).update({"backend": "cpu", "language": "c", "device": "cpu"})
    report = {
        "mode": "existing_cpu_only",
        "backend": "cpu",
        "shape": {key: ir["problem"][key] for key in ("M", "N", "K")},
        "kernel": kernel,
        "max_candidates": max_candidates,
        "rounds": [],
        "results": [],
        "source_hash": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "target_ratio": target_ratio,
        "target_reached": False,
        "benchmark_runs": benchmark_runs,
        "warmup_runs": warmup_runs,
        "aggregation": aggregation,
    }
    seen: set[str] = set()
    uses_native_verifier = verify is verify_cpu_build_and_run
    cached_baseline_result = None
    if uses_native_verifier:
        baseline_dir = output / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        cached_baseline_result = run_cpu_optimized_baseline(
            baseline,
            baseline_dir,
            timeout,
            openblas_root,
            platform,
        )
        report["baseline_cache"] = {
            "enabled": True,
            "status": cached_baseline_result.get("status"),
            "metrics": cached_baseline_result.get("metrics", {}),
        }

    def evaluate(content: str, tile: dict[str, int] | None, round_number: int) -> bool:
        candidate_started = time.perf_counter()
        fingerprint = hashlib.sha256(content.encode()).hexdigest()
        if fingerprint in seen:
            return False
        seen.add(fingerprint)
        number = len(report["results"])
        folder = output / f"candidate_{number:04d}"
        folder.mkdir()
        for name, text in {**sources, kernel: content}.items():
            (folder / name).write_text(text, encoding="utf-8")
        state = clear_results(baseline)
        if tile:
            state.setdefault("cpu_tiling", {}).update({
                "l2_block_m": tile["L2_M"], "l2_block_n": tile["L2_N"], "l2_block_k": tile["L2_K"],
                "l1_block_m": tile["L1_M"], "l1_block_n": tile["L1_N"], "l1_block_k": tile["L1_K"],
                "register_m": tile["MR"], "register_n": tile["NR"],
            })
        state["specialization"] = {
            "executor": "existing_cpu_source_parameters",
            "tile": tile,
            "source_hash": report["source_hash"],
        }
        print(f"[{report['shape']}] backend=cpu candidate={number} round={round_number} tile={tile}", flush=True)
        try:
            verify_kwargs = {
                "source_dir": folder,
                "build_dir": folder / "build",
                "build_platform": platform,
                "timeout_seconds": timeout,
                "benchmark_runs": max(1, int(screening_benchmark_runs)),
                "benchmark_warmup_runs": max(0, int(screening_warmup_runs)),
                "openblas_root": openblas_root,
            }
            if uses_native_verifier:
                verify_kwargs["precomputed_baseline_result"] = cached_baseline_result
            result = verify(state, **verify_kwargs)
        except Exception as exc:
            result = state
            result["verification"] = {"accepted": False, "error_type": type(exc).__name__, "error": str(exc)}
        verification = result.get("verification", {}) or {}
        selected_gflops = aggregate_value(result, "gflops", aggregation) if passed(result) else None
        cpu_blas = (result.get("performance", {}) or {}).get("cpu_blas_gflops")
        record = {
            "candidate_id": number,
            "round": round_number,
            "code_dir": str(folder),
            "tile": tile,
            "source_hash": fingerprint,
            "accepted": passed(result),
            "compile_status": verification.get("compile", {}).get("status", "not_run"),
            "correctness_status": verification.get("correctness", {}).get("status", "not_run"),
            "runtime_safety_status": verification.get("runtime_safety", {}).get("status", "not_run"),
            "gflops": selected_gflops,
            "latency_ms": aggregate_value(result, "latency_ms", aggregation),
            "cpu_blas_gflops": cpu_blas,
            "relative_to_cpu_blas": selected_gflops / cpu_blas if selected_gflops and cpu_blas else None,
            "performance": result.get("performance", {}),
            "evaluation_seconds": round(time.perf_counter() - candidate_started, 6),
        }
        write_json(folder / "verified_ir.json", result)
        write_json(folder / "candidate.json", record)
        report["results"].append(record)
        write_json(output / "search.json", report)
        print(f"  compile={record['compile_status']} correct={record['correctness_status']} GFLOPS={record['gflops']}", flush=True)
        return True

    evaluate(sources[kernel], None, 0)
    if candidate_tiles is None:
        proposed_tiles, search_space_audit = build_cpu_tile_space(library, sources[kernel], baseline)
    else:
        proposed_tiles = candidate_tiles
        search_space_audit = {"source": "explicit_candidate_tiles", "accepted_candidate_count": len(candidate_tiles)}
    report["search_space_audit"] = search_space_audit
    all_tiles = []
    materialized_hashes = {hashlib.sha256(sources[kernel].encode()).hexdigest()}
    current_cpu_tile = extract_cpu_tile(sources[kernel])
    register_tunable = "SCOPE_CPU_TUNABLE_REGISTER_TILE" in sources[kernel]
    for proposed_tile in proposed_tiles:
        tile = dict(proposed_tile)
        if not register_tunable:
            for key in ("MR", "NR"):
                if key in current_cpu_tile:
                    tile[key] = current_cpu_tile[key]
        content = specialize_cpu_source(sources[kernel], tile)
        if content is None:
            continue
        fingerprint = hashlib.sha256(content.encode()).hexdigest()
        if fingerprint in materialized_hashes:
            continue
        materialized_hashes.add(fingerprint)
        all_tiles.append(tile)
    report["proposed_candidate_count"] = len(proposed_tiles)
    report["legal_candidate_count"] = len(all_tiles)
    report["candidate_order"] = "L2 -> L1 -> register tile"
    report["frozen_parameters"] = (
        ["MR", "NR"]
        if "SCOPE_CPU_TUNABLE_REGISTER_TILE" not in sources[kernel]
        else []
    )
    pool = list(all_tiles if max_candidates == 0 else all_tiles[:max_candidates])
    report["budget_omitted_count"] = len(all_tiles) - len(pool)
    write_json(output / "candidate_manifest.json", all_tiles)
    per_round = max(1, (len(pool) + rounds - 1) // rounds) if pool else 1
    pending = list(pool)
    for round_number in range(1, rounds + 1):
        before = len(report["results"])
        attempted = 0
        while pending and attempted < per_round:
            tile = pending.pop(0)
            attempted += 1
            content = specialize_cpu_source(sources[kernel], tile)
            if content is not None:
                evaluate(content, tile, round_number)
        report["rounds"].append({"round": round_number, "tested": len(report["results"]) - before})
        if target_ratio is not None and any(
            (item.get("relative_to_cpu_blas") or 0) >= target_ratio for item in report["results"]
        ):
            report.update(target_reached=True, stop_reason="target_reached")
            break
        if not pending:
            report["stop_reason"] = "budget_exhausted" if report["budget_omitted_count"] else "candidate_pool_exhausted"
            break
    report.setdefault("stop_reason", "budget_exhausted")
    preliminary_top = sorted(
        (item for item in report["results"] if item["accepted"] and item["gflops"] is not None),
        key=lambda item: item["gflops"],
        reverse=True,
    )[:max(3, int(final_remeasure_top_k))]
    if uses_native_verifier:
        for record in preliminary_top[:max(0, int(final_remeasure_top_k))]:
            folder = Path(record["code_dir"])
            state = clear_results(baseline)
            if record.get("tile"):
                tile = record["tile"]
                state.setdefault("cpu_tiling", {}).update({
                    "l2_block_m": tile["L2_M"], "l2_block_n": tile["L2_N"], "l2_block_k": tile["L2_K"],
                    "l1_block_m": tile["L1_M"], "l1_block_n": tile["L1_N"], "l1_block_k": tile["L1_K"],
                    "register_m": tile["MR"], "register_n": tile["NR"],
                })
            measured = verify(
                state,
                source_dir=folder,
                build_dir=folder / "build_final",
                build_platform=platform,
                timeout_seconds=timeout,
                benchmark_runs=benchmark_runs,
                benchmark_warmup_runs=warmup_runs,
                openblas_root=openblas_root,
                precomputed_baseline_result=cached_baseline_result,
            )
            verification = measured.get("verification", {}) or {}
            record.update({
                "accepted": passed(measured),
                "compile_status": verification.get("compile", {}).get("status", "not_run"),
                "correctness_status": verification.get("correctness", {}).get("status", "not_run"),
                "runtime_safety_status": verification.get("runtime_safety", {}).get("status", "not_run"),
                "gflops": aggregate_value(measured, "gflops", aggregation) if passed(measured) else None,
                "latency_ms": aggregate_value(measured, "latency_ms", aggregation),
                "performance": measured.get("performance", {}),
                "measurement_phase": "final_remeasure",
            })
            cpu_blas = (measured.get("performance", {}) or {}).get("cpu_blas_gflops")
            record["cpu_blas_gflops"] = cpu_blas
            record["relative_to_cpu_blas"] = record["gflops"] / cpu_blas if record["gflops"] and cpu_blas else None
            write_json(folder / "verified_ir.json", measured)
            write_json(folder / "candidate.json", record)
    top = sorted(
        (item for item in preliminary_top if item["accepted"] and item["gflops"] is not None),
        key=lambda item: item["gflops"],
        reverse=True,
    )[:3]
    for rank, record in enumerate(top, 1):
        record["rank"] = rank
        destination = output / "top_3" / f"rank_{rank}"
        destination.mkdir(parents=True, exist_ok=True)
        original = Path(record["code_dir"])
        for name in (*sources.keys(), "verified_ir.json", "candidate.json"):
            shutil.copyfile(original / name, destination / name)
    report["top_results"] = top
    report["tested_candidate_count"] = len(report["results"])
    report["accepted_candidate_count"] = sum(bool(item.get("accepted")) for item in report["results"])
    report["failed_candidate_count"] = report["tested_candidate_count"] - report["accepted_candidate_count"]
    report["compile_attempt_count"] = report["tested_candidate_count"] + (
        min(len(preliminary_top), max(0, int(final_remeasure_top_k))) if uses_native_verifier else 0
    )
    report["elapsed_seconds"] = round(time.perf_counter() - tuning_started, 6)
    write_json(output / "top_3_results.json", top)
    lines = [
        f"# CPU GEMM {report['shape']}", "", f"Stop reason: {report['stop_reason']}", "",
        "| Rank | L2/L1/Register tile | GFLOPS | Latency ms | / OpenBLAS |",
        "|---|---|---|---|---|",
    ]
    for record in top:
        lines.append(
            f"| {record['rank']} | {record['tile'] or 'original source'} | {record['gflops']} | "
            f"{record['latency_ms']} | {record['relative_to_cpu_blas']} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "search.json", report)
    return report


def tune_sources(*args: Any, backend: str = "cuda", **kwargs: Any) -> dict[str, Any]:
    if backend == "cpu":
        return tune_cpu_sources(*args, **kwargs)
    return tune_cuda_sources(*args, **kwargs)


def _publish_gpu_shape_bundle(
    output: Path,
    sources: dict[str, str],
    shape_items: list[dict[str, Any]],
) -> Path:
    result_dir = output / "results"
    result_dir.mkdir(parents=True, exist_ok=False)
    sizes = [int(item["shape"]["M"]) for item in shape_items]
    kernel_text_by_size: dict[int, str] = {}
    for item, size in zip(shape_items, sizes):
        kernel_text = (Path(item["final_code_dir"]) / "cuda_kernel.cuh").read_text(encoding="utf-8")
        kernel_text_by_size[size] = kernel_text
        (result_dir / f"cuda_kernel_{size}.cuh").write_text(kernel_text, encoding="utf-8")
    (result_dir / "main.cpp").write_text(sources["main.cpp"], encoding="utf-8")
    # A single-candidate header may include cuda_kernel.cuh directly. The
    # integrated bundle owns all numbered kernel includes in kernel_dispatch.cu.
    kernel_header = re.sub(
        r'^\s*#\s*include\s*[<"]cuda_kernel\.cuh[>"]\s*\r?\n?',
        "",
        sources["kernel.h"],
        flags=re.MULTILINE,
    )
    (result_dir / "kernel.h").write_text(kernel_header, encoding="utf-8")

    declarations = ["#pragma once", ""]
    for size in sizes:
        declarations.extend([
            f"void cuda_gemm_{size}(",
            "    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);",
        ])
    (result_dir / "kernel_variants.h").write_text("\n".join(declarations) + "\n", encoding="utf-8")

    function_pattern = re.compile(
        r'(?m)^\s*(?:extern\s+"C"\s+)?'
        r'(?:(?:__global__|__device__|__host__|__forceinline__|inline|static)\s+)*'
        r'(?:void|int|float|double|bool|cudaError_t|dim3)\s+'
        r'([A-Za-z_]\w*)\s*\('
    )
    names_by_size = {
        size: set(function_pattern.findall(kernel_text))
        for size, kernel_text in kernel_text_by_size.items()
    }
    duplicate_helpers = sorted(
        name for name in set().union(*names_by_size.values())
        if name not in {"gemm", "cuda_gemm"}
        and sum(name in names for names in names_by_size.values()) > 1
    )

    dispatch = ['#include "kernel.h"', '#include "kernel_variants.h"', ""]
    for size in sizes:
        renamed = [("gemm", f"scope_gemm_{size}"), ("cuda_gemm", f"cuda_gemm_{size}")]
        renamed.extend(
            (name, f"{name}_{size}")
            for name in duplicate_helpers
            if name in names_by_size[size]
        )
        dispatch.extend(f"#define {name} {replacement}" for name, replacement in renamed)
        dispatch.append(f'#include "cuda_kernel_{size}.cuh"')
        dispatch.extend(f"#undef {name}" for name, _ in reversed(renamed))
        dispatch.append("")
    dispatch.extend([
        "static int maximum_dimension(int M, int N, int K) {",
        "    int value = M > N ? M : N;",
        "    return value > K ? value : K;",
        "}",
        "",
        "void cuda_gemm(",
        "    int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {",
        "    const int extent = maximum_dimension(M, N, K);",
    ])
    for index, size in enumerate(sizes):
        keyword = "if" if index == 0 else "else if"
        if index < len(sizes) - 1:
            dispatch.extend([
                f"    {keyword} (extent <= {size}) {{",
                f"        cuda_gemm_{size}(M, N, K, alpha, A, B, beta, C);",
                "    }",
            ])
        else:
            if index == 0:
                dispatch.append(f"    cuda_gemm_{size}(M, N, K, alpha, A, B, beta, C);")
            else:
                dispatch.extend([
                    "    else {",
                    f"        cuda_gemm_{size}(M, N, K, alpha, A, B, beta, C);",
                    "    }",
                ])
    dispatch.append("}")
    (result_dir / "kernel_dispatch.cu").write_text("\n".join(dispatch) + "\n", encoding="utf-8")

    support_candidates = (
        ROOT / "4060ti" / "results",
        ROOT / "4060ti" / "qwen3.5-397b-a17b",
        ROOT / "4060ti" / "deepseek-v4.1-flash",
        ROOT / "4060ti" / "GLM-5.2",
        ROOT / "H800" / "SCOPE",
    )
    support_root = next(
        (candidate for candidate in support_candidates if (candidate / "Makefile").is_file()),
        None,
    )
    if support_root is None:
        checked = ", ".join(str(candidate / "Makefile") for candidate in support_candidates)
        raise FileNotFoundError(f"GPU bundle Makefile missing; checked: {checked}")
    makefile_source = support_root / "Makefile"
    makefile_text = makefile_source.read_text(encoding="utf-8")
    dependency_names = " ".join(f"cuda_kernel_{size}.cuh" for size in sizes)
    makefile_text = re.sub(
        r"^DEPS\s*:=.*$",
        f"DEPS := kernel.h kernel_variants.h {dependency_names}",
        makefile_text,
        count=1,
        flags=re.MULTILINE,
    )
    makefile_text = re.sub(
        r"^\.PHONY:.*$",
        ".PHONY: all clean info " + " ".join(f"run-{size}" for size in sizes),
        makefile_text,
        count=1,
        flags=re.MULTILINE,
    )
    makefile_text = re.sub(
        r"(?m)^run-\d+: all\r?\n\t.*(?:\r?\n|$)",
        "",
        makefile_text,
    ).rstrip()
    run_targets = "\n".join(
        f"run-{size}: all\n\t./$(TARGET_FILE) {size} {size} {size}"
        for size in sizes
    )
    (result_dir / "Makefile").write_text(
        makefile_text + "\n\n" + run_targets + "\n",
        encoding="utf-8",
    )
    windows_build = support_root / "build_windows.bat"
    if windows_build.is_file():
        shutil.copyfile(windows_build, result_dir / "build_windows.bat")
    return result_dir


def _publish_cpu_shape_bundle(
    output: Path,
    sources: dict[str, str],
    shape_items: list[dict[str, Any]],
) -> Path:
    result_dir = output / "results"
    result_dir.mkdir(parents=True, exist_ok=False)
    sizes = [int(item["shape"]["M"]) for item in shape_items]
    for item, size in zip(shape_items, sizes):
        shutil.copyfile(Path(item["final_code_dir"]) / "cpu_kernel.c", result_dir / f"cpu_kernel_{size}.c")
    for name in ("main.c", "kernel.h"):
        (result_dir / name).write_text(sources[name], encoding="utf-8")

    declarations = ["#ifndef SCOPE_CPU_KERNEL_VARIANTS_H", "#define SCOPE_CPU_KERNEL_VARIANTS_H", ""]
    for size in sizes:
        declarations.extend([
            f"void cpu_gemm_{size}(",
            "    int M, int N, int K, float alpha,",
            "    const float *A, const float *B, float beta, float *C);",
        ])
    declarations.extend(["", "#endif"])
    (result_dir / "kernel_variants.h").write_text("\n".join(declarations) + "\n", encoding="utf-8")

    dispatch = [
        '#include "kernel.h"',
        '#include "kernel_variants.h"',
        "",
        "static int maximum_dimension(int M, int N, int K) {",
        "    int value = M > N ? M : N;",
        "    return value > K ? value : K;",
        "}",
        "",
        "void cpu_gemm(",
        "    int M, int N, int K, float alpha,",
        "    const float *A, const float *B, float beta, float *C) {",
        "    const int extent = maximum_dimension(M, N, K);",
    ]
    for index, size in enumerate(sizes):
        keyword = "if" if index == 0 else "else if"
        if index < len(sizes) - 1:
            dispatch.extend([
                f"    {keyword} (extent <= {size}) {{",
                f"        cpu_gemm_{size}(M, N, K, alpha, A, B, beta, C);",
                "    }",
            ])
        else:
            if index == 0:
                dispatch.append(f"    cpu_gemm_{size}(M, N, K, alpha, A, B, beta, C);")
            else:
                dispatch.extend([
                    "    else {",
                    f"        cpu_gemm_{size}(M, N, K, alpha, A, B, beta, C);",
                    "    }",
                ])
    dispatch.append("}")
    (result_dir / "kernel_dispatch.c").write_text("\n".join(dispatch) + "\n", encoding="utf-8")

    object_names = ["main", "kernel_dispatch", *(f"cpu_kernel_{size}" for size in sizes)]
    makefile_lines = [
        "TARGET ?= gemm_cpu",
        "BUILD_DIR ?= build",
        "PYTHON ?= python",
        ".DEFAULT_GOAL := all",
        "",
        "ifeq ($(OS),Windows_NT)",
        "EXEEXT := .exe",
        "MKDIR_BUILD = $(PYTHON) -c \"from pathlib import Path; Path(r'$(BUILD_DIR)').mkdir(parents=True, exist_ok=True)\"",
        "REMOVE_BUILD = $(PYTHON) -c \"import shutil; shutil.rmtree(r'$(BUILD_DIR)', ignore_errors=True)\"",
        "else",
        "EXEEXT :=",
        "MKDIR_BUILD = mkdir -p \"$(BUILD_DIR)\"",
        "REMOVE_BUILD = rm -rf \"$(BUILD_DIR)\"",
        "endif",
        "",
        "ifeq ($(origin CC),default)",
        "CC := gcc",
        "endif",
        "TARGET_FILE := $(BUILD_DIR)/$(TARGET)$(EXEEXT)",
        "CFLAGS ?= -O3 -std=c11 -march=native -ffast-math -funroll-loops",
        "CFLAGS += -mavx2 -mfma -fopenmp",
        "LDLIBS ?= -lm -fopenmp",
        "OBJECTS := " + " ".join(f"$(BUILD_DIR)/{name}.o" for name in object_names),
        "",
        ".PHONY: all clean info " + " ".join(f"run-{size}" for size in sizes),
        "all: $(TARGET_FILE)",
        "info:",
        "\t@echo CC=$(CC)",
        "\t@echo TARGET=$(TARGET_FILE)",
        "$(BUILD_DIR):",
        "\t$(MKDIR_BUILD)",
        "$(BUILD_DIR)/main.o: main.c kernel.h | $(BUILD_DIR)",
        "\t$(CC) $(CFLAGS) -c $< -o $@",
        "$(BUILD_DIR)/kernel_dispatch.o: kernel_dispatch.c kernel.h kernel_variants.h | $(BUILD_DIR)",
        "\t$(CC) $(CFLAGS) -c $< -o $@",
    ]
    for size in sizes:
        makefile_lines.extend([
            f"$(BUILD_DIR)/cpu_kernel_{size}.o: cpu_kernel_{size}.c kernel.h | $(BUILD_DIR)",
            f"\t$(CC) $(CFLAGS) -Dcpu_gemm=cpu_gemm_{size} -c $< -o $@",
        ])
    makefile_lines.extend([
        "$(TARGET_FILE): $(OBJECTS)",
        "\t$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDLIBS)",
    ])
    for size in sizes:
        makefile_lines.extend([
            f"run-{size}: all",
            f"\t./$(TARGET_FILE) {size} {size} {size}",
        ])
    makefile_lines.extend(["clean:", "\t$(REMOVE_BUILD)"])
    (result_dir / "Makefile").write_text("\n".join(makefile_lines) + "\n", encoding="utf-8")
    return result_dir


def publish_tuned_shape_bundle(
    output: Path,
    sources: dict[str, str],
    shape_items: list[dict[str, Any]],
    backend: str,
) -> Path:
    actual = tuple(int(item["shape"]["M"]) for item in shape_items)
    square = all(
        item["shape"]["M"] == item["shape"]["N"] == item["shape"]["K"]
        for item in shape_items
    )
    if not actual or len(set(actual)) != len(actual) or not square:
        raise ValueError(f"Integrated {backend} bundle requires unique square shapes, got {actual}")
    if any(item.get("status") != "completed" for item in shape_items):
        raise ValueError("Integrated bundle requires one verified Top-1 for every target shape")
    result_dir = (
        _publish_cpu_shape_bundle(output, sources, shape_items)
        if backend == "cpu"
        else _publish_gpu_shape_bundle(output, sources, shape_items)
    )
    write_json(result_dir / "shape_manifest.json", {
        "backend": backend,
        "shapes": shape_items,
    })
    return result_dir


def tune_top1_shapes(
    seed: dict[str, Any] | None,
    library: dict[str, Any],
    output: Path,
    config: dict[str, Any],
    *,
    backend: str,
    platform: str,
    benchmark_runs: int,
    warmup_runs: int,
    openblas_root: Path | None = None,
) -> dict[str, Any]:
    """Expand one verified terminal winner into independently tuned shape winners."""
    phase_started = time.perf_counter()
    if backend not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported tuning backend: {backend}")
    output = Path(output).resolve() / uuid4().hex[:10]
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "stage": "FinalTop1ShapeTuning",
        "backend": backend,
        "output_dir": str(output),
        "seed_code_dir": seed.get("candidate_code_dir") if seed else None,
        "shapes": [],
    }
    if not seed or not seed.get("accepted") or not passed(seed.get("verified_ir", {})):
        report["status"] = "skipped_no_verified_top1"
        report["elapsed_seconds"] = round(time.perf_counter() - phase_started, 6)
        write_json(output / "summary.json", report)
        return report

    kernel = "cpu_kernel.c" if backend == "cpu" else "cuda_kernel.cuh"
    if seed.get("source_snapshot"):
        sources = copy.deepcopy(seed["source_snapshot"])
    else:
        _, kernel, sources = read_sources(Path(seed["candidate_code_dir"]), backend)
    expected_harness = "main.c" if backend == "cpu" else "main.cpp"
    if kernel not in sources or expected_harness not in sources:
        raise ValueError(f"Top-1 source snapshot must contain {kernel} and {expected_harness}")

    default_shapes = (256, 512, 768, 1024) if backend == "cpu" else (512, 1024, 2048, 4096)
    configured_shapes = config.get("cpu_shapes" if backend == "cpu" else "gpu_shapes", default_shapes)
    shapes: list[tuple[int, int, int]] = []
    for value in configured_shapes:
        shape = (value, value, value) if isinstance(value, int) else tuple(value)
        if len(shape) != 3 or any(not isinstance(item, int) or item <= 0 for item in shape):
            raise ValueError(f"Final tuning shape must be a positive integer or M/N/K triple: {value}")
        if shape not in shapes:
            shapes.append(shape)

    max_candidates = max(0, int(config.get("max_candidates_per_shape", 32)))
    rounds = max(1, int(config.get("rounds", 3)))
    timeout = max(1, int(config.get("timeout_seconds", 180)))
    aggregation = str(config.get("aggregation", "trimmed_mean"))
    if aggregation not in {"mean", "trimmed_mean"}:
        raise ValueError("final_top1_tuning.aggregation must be mean or trimmed_mean")
    target_ratio = config.get("target_ratio")
    screening_benchmark_runs = max(1, int(config.get("screening_benchmark_runs", 1)))
    screening_warmup_runs = max(0, int(config.get("screening_warmup_runs", 0)))
    final_remeasure_top_k = max(0, int(config.get("final_remeasure_top_k", 3)))
    padding_modes = tuple(config.get("padding_modes", ["original"]))

    seed_ir = clear_results(seed["verified_ir"])
    report["seed_gflops"] = aggregate_value(seed["verified_ir"], "gflops", aggregation)
    report["max_candidates_per_shape"] = max_candidates
    report["aggregation"] = aggregation
    for shape in shapes:
        shape_started = time.perf_counter()
        shape_name = "x".join(map(str, shape))
        tuning_dir = output / shape_name / "search"
        item: dict[str, Any] = {
            "shape": dict(zip(("M", "N", "K"), shape)),
            "tuning_dir": str(tuning_dir),
        }
        try:
            ir = clear_results(seed_ir)
            ir.setdefault("problem", {}).update(item["shape"])
            common = {
                "max_candidates": max_candidates,
                "rounds": rounds,
                "benchmark_runs": benchmark_runs,
                "warmup_runs": warmup_runs,
                "platform": platform,
                "timeout": timeout,
                "target_ratio": target_ratio,
                "aggregation": aggregation,
            }
            if backend == "cpu":
                common["openblas_root"] = openblas_root
                common["screening_benchmark_runs"] = screening_benchmark_runs
                common["screening_warmup_runs"] = screening_warmup_runs
                common["final_remeasure_top_k"] = final_remeasure_top_k
            else:
                common["padding_modes"] = padding_modes
                common["max_variant_compilations"] = max_candidates or None
            tuned = tune_sources(
                copy.deepcopy(sources), kernel, ir, library, tuning_dir,
                backend=backend, **common,
            )
            winners = [
                result for result in tuned.get("top_results", [])
                if result.get("accepted") and result.get("gflops") is not None
            ]
            seed_result = next(
                (result for result in tuned.get("results", []) if result.get("candidate_id") == 0),
                {},
            )
            tested = int(tuned.get("tested_candidate_count", len(tuned.get("results", []))))
            accepted = int(tuned.get("accepted_candidate_count", sum(
                bool(result.get("accepted")) for result in tuned.get("results", [])
            )))
            item["seed_gflops"] = seed_result.get("gflops")
            item["search_cost"] = {
                "tested_candidates": tested,
                "accepted_candidates": accepted,
                "failed_candidates": int(tuned.get("failed_candidate_count", tested - accepted)),
                "compile_attempts": int(tuned.get("compile_attempt_count", tested)),
                "benchmark_process_runs": estimate_tuning_benchmark_process_runs(
                    tuned, backend, benchmark_runs, screening_benchmark_runs,
                    final_remeasure_top_k,
                ),
                "rounds_executed": len(tuned.get("rounds", [])),
                "legal_candidates": tuned.get("legal_candidate_count"),
                "budget_omitted_candidates": tuned.get("budget_omitted_count"),
                "elapsed_seconds": tuned.get("elapsed_seconds"),
            }
            if not winners:
                item.update(status="no_verified_result", stop_reason=tuned.get("stop_reason"))
            else:
                winner = max(winners, key=lambda result: float(result["gflops"]))
                source_dir = Path(winner["code_dir"])
                final_dir = output / shape_name / "top_1"
                final_dir.mkdir(parents=True, exist_ok=False)
                copied = []
                for name in (*sources.keys(), "verified_ir.json", "candidate.json"):
                    source_file = source_dir / name
                    if source_file.is_file():
                        shutil.copyfile(source_file, final_dir / name)
                        copied.append(name)
                item.update(
                    status="completed",
                    gflops=winner.get("gflops"),
                    latency_ms=winner.get("latency_ms"),
                    relative_to_cublas=winner.get("relative_to_cublas"),
                    relative_to_cpu_blas=winner.get("relative_to_cpu_blas"),
                    tile=winner.get("tile"),
                    padding_mode=winner.get("padding_mode"),
                    final_code_dir=str(final_dir),
                    copied_files=copied,
                    stop_reason=tuned.get("stop_reason"),
                )
        except Exception as exc:
            item.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        item["wallclock_seconds"] = round(time.perf_counter() - shape_started, 6)
        report["shapes"].append(item)
        write_json(output / "summary.json", report)
    completed = sum(item.get("status") == "completed" for item in report["shapes"])
    report["completed_shape_count"] = completed
    report["status"] = (
        "completed" if completed == len(report["shapes"])
        else "partial_failure" if completed
        else "failed"
    )
    report["elapsed_seconds"] = round(time.perf_counter() - phase_started, 6)
    if report["status"] == "completed":
        try:
            result_dir = publish_tuned_shape_bundle(output, sources, report["shapes"], backend)
            report["published_results_dir"] = str(result_dir)
            report["publication_status"] = "completed"
        except Exception as exc:
            report["publication_status"] = "failed"
            report["publication_error_type"] = type(exc).__name__
            report["publication_error"] = str(exc)
            report["status"] = "partial_failure"
    else:
        report["publication_status"] = "skipped_incomplete_shapes"
    write_json(output / "summary.json", report)
    return report


def estimate_tuning_benchmark_process_runs(
    tuned: dict[str, Any],
    backend: str,
    final_runs: int,
    screening_runs: int,
    final_remeasure_top_k: int,
) -> int:
    tested = int(tuned.get("tested_candidate_count", len(tuned.get("results", []))))
    if backend == "cpu":
        remeasured = min(
            int(tuned.get("accepted_candidate_count", 0)),
            max(0, int(final_remeasure_top_k)),
        )
        return tested * max(1, int(screening_runs)) + remeasured * max(1, int(final_runs))
    runs = int(tuned.get("benchmark_runs", final_runs) or final_runs)
    return tested * max(1, runs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source", type=Path, required=True)
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument("--shapes", type=int, nargs="+")
    shapes.add_argument("--matrix-size", type=int, nargs=3, metavar=("M", "N", "K"))
    parser.add_argument("--ir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "conf.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-platform", choices=("windows", "linux"), default="windows")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--aggregation", choices=("mean", "trimmed_mean"), default="mean")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--target-ratio", type=float)
    args = parser.parse_args(argv)
    sizes = [args.matrix_size] if args.matrix_size else [[n, n, n] for n in (args.shapes or [1024])]
    if any(min(shape) <= 0 for shape in sizes) or args.max_candidates < 0:
        parser.error("Sizes must be positive and max-candidates must be nonnegative")
    source_dir, kernel, sources = read_sources(args.source, args.backend)
    output = (args.output or ROOT / "gemm_code" / "shape_tuning" / uuid4().hex[:10]).resolve()
    if output == source_dir or source_dir in output.parents or output in source_dir.parents:
        parser.error("Output must be separate from the source directory")
    template_path = args.ir or ROOT / "data" / "IRs" / "optir.json"
    template = json.loads(template_path.read_text(encoding="utf-8-sig"))
    description = f"FP32 row-major NN GEMM on {args.backend.upper()}, matrix size 1024x1024x1024"
    base = build_extracted_ir(template, description)
    base.setdefault("target", {}).update(
        {"backend": "cpu", "language": "c", "device": "cpu"}
        if args.backend == "cpu" else
        {"backend": "cuda", "language": "cuda_cpp", "device": "gpu"}
    )
    library_name = "cpu_strategy_library.json" if args.backend == "cpu" else "strategy_library.json"
    library = json.loads((ROOT / "data" / "lib" / library_name).read_text(encoding="utf-8"))
    config = load_config(args.config)
    openblas = (config.get("cpu_baseline", {}) or {}).get("openblas_root")
    reports = []
    published_shapes: list[dict[str, Any]] = []
    for shape in sizes:
        ir = copy.deepcopy(base)
        ir["problem"].update(zip(("M", "N", "K"), shape))
        common = dict(
            max_candidates=args.max_candidates,
            rounds=args.rounds,
            platform=args.build_platform,
            benchmark_runs=args.benchmark_runs,
            warmup_runs=args.warmup_runs,
            timeout=args.timeout,
            target_ratio=args.target_ratio,
            aggregation=args.aggregation,
        )
        if args.backend == "cpu":
            common["openblas_root"] = Path(openblas) if openblas else None
        result = tune_sources(
            sources, kernel, ir, library, output / "x".join(map(str, shape)),
            backend=args.backend, **common,
        )
        reports.append({"backend": args.backend, "shape": shape, "stop_reason": result["stop_reason"], "top_results": result["top_results"]})
        verified = [
            candidate for candidate in result.get("top_results", [])
            if candidate.get("accepted") and candidate.get("gflops") is not None
        ]
        if verified:
            winner = max(verified, key=lambda candidate: float(candidate["gflops"]))
            published_shapes.append({
                "shape": dict(zip(("M", "N", "K"), shape)),
                "status": "completed",
                "gflops": winner.get("gflops"),
                "latency_ms": winner.get("latency_ms"),
                "tile": winner.get("tile"),
                "padding_mode": winner.get("padding_mode"),
                "final_code_dir": winner["code_dir"],
            })
    write_json(output / "summary.json", reports)
    publication: dict[str, Any] = {
        "status": "skipped_no_verified_result",
        "requested_shape_count": len(sizes),
        "published_shape_count": len(published_shapes),
    }
    if published_shapes:
        try:
            result_dir = publish_tuned_shape_bundle(output, sources, published_shapes, args.backend)
            publication.update(status="completed", results_dir=str(result_dir))
        except Exception as exc:
            publication.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
    write_json(output / "publication.json", publication)
    print(f"Results: {output / 'summary.json'}", flush=True)
    if publication.get("results_dir"):
        print(f"Build bundle: {publication['results_dir']}", flush=True)


if __name__ == "__main__":
    main()
