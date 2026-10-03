"""Persistent implementations plus bounded, LLM-free shape specialization.

Built-in SIMT variants are explicitly identified: they never claim to preserve
the seed's pipeline/vectorization. The seed is independently retested unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

from SCOPE.generate_ir.tiling_planner import build_joint_tiling_candidates, make_diverse_tiling_pool
from SCOPE.generate_ir.gpu_resources import hardware_value
from SCOPE.verification.build_run_verifier import verify_build_and_run

ROOT = Path(__file__).resolve().parents[1]
TILE_KEYS = ("BM", "BN", "BK", "WM", "WN", "TM", "TN", "WMITER", "WNITER")
SEMANTIC_KEYS = ("dtype_A", "dtype_B", "dtype_C", "accum_dtype", "layout_A", "layout_B", "layout_C", "trans_A", "trans_B", "operation", "alpha", "beta")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def identity(ir):
    hw = ir.get("hardware", {})
    gpu = hw.get("gpu") or hw
    return {
        "gpu_name": gpu.get("gpu_name") or gpu.get("name") or hw.get("gpu_name"),
        "compute_capability": gpu.get("compute_capability") or hw.get("compute_capability"),
        "sm_count": gpu.get("sm_count") or hw.get("sm_count"),
        "semantics": {key: ir.get("problem", {}).get(key) for key in SEMANTIC_KEYS},
    }


def passed(ir):
    v = ir.get("verification", {})
    return v.get("accepted") is True and all(v.get(k, {}).get("status") == "pass"
                                             for k in ("compile", "correctness", "runtime_safety"))


def metric(ir, prefix=""):
    perf = ir.get("performance", {})
    for key in (prefix+"gflops_trimmed_mean", prefix+"gflops_mean", prefix+"gflops"):
        value = perf.get(key)
        if isinstance(value, (float, int)) and math.isfinite(value) and value > 0:
            return float(value)
    return None


def ratio(ir):
    own, reference = metric(ir), metric(ir, "cublas_")
    return own / reference if own and reference else None


def archive(candidate, registry):
    """Archive source and verified evidence; no executables or API credentials."""
    ir = candidate.get("verified_ir", {})
    sources = candidate.get("source_snapshot", {})
    if not passed(ir) or "cuda_kernel.cuh" not in sources or not identity(ir)["gpu_name"]:
        return None
    sources = {name: content for name, content in sources.items()
               if Path(name).name == name and Path(name).suffix in {".cuh", ".h", ".cu", ".cpp"}}
    digest = hashlib.sha256(json.dumps([identity(ir), sources], sort_keys=True).encode()).hexdigest()[:20]
    folder = Path(registry) / digest
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in sources.items():
        (folder / name).write_text(content, encoding="utf-8")
    write_json(folder / "family.json", {"family_id": digest, "identity": identity(ir),
        "verified_ir": ir, "source_files": list(sources), "source_hash": digest,
        "provenance": candidate.get("source_phase", "scope_generation"),
        "validation_scope": "archived shape only; all new instances must be retested"})
    return digest


def load_families(registry, ir):
    result = []
    if not identity(ir)["gpu_name"]:
        return result
    for path in sorted(Path(registry).glob("*/family.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["identity"] != identity(ir) or not passed(record["verified_ir"]):
                continue
            sources = {name: (path.parent / name).read_text(encoding="utf-8")
                       for name in record["source_files"] if Path(name).name == name}
            digest = hashlib.sha256(json.dumps([record["identity"], sources], sort_keys=True).encode()).hexdigest()[:20]
            if digest != record["source_hash"]:
                continue
            result.append({**record, "sources": sources})
        except (OSError, ValueError, KeyError):
            continue
    return result


def shared_bytes(tile, transpose_a, padding):
    bm, bn, bk = (tile[k] for k in ("BM", "BN", "BK"))
    return 4*((bk*(bm+padding) if transpose_a else bm*(bk+padding)) + bk*(bn+padding))


def render(tile, transpose_a=False, padding=0, unroll=1, split_k=1, stages=1, async_copy=False):
    params = {**{key: int(tile[key]) for key in TILE_KEYS}, "TRANSPOSE_A": int(transpose_a),
              "PADDING": padding, "UNROLL": unroll, "SPLIT_K": split_k,
              "STAGES": stages, "ASYNC": int(async_copy)}
    prefix = "\n".join(f"#define SCOPE_{key} {value}" for key, value in params.items())
    return prefix + "\n" + Path(__file__).with_name("simt_family.cuh").read_text(encoding="utf-8")


def instance_ir(base, tile, transpose_a, padding, unroll, split_k, stages=1, async_copy=False):
    # Do not inherit claims or verification results from the archived program.
    ir = {key: copy.deepcopy(base[key]) for key in ("problem", "hardware") if key in base}
    bm, bn, bk = (tile[k] for k in ("BM", "BN", "BK"))
    ir["tiling"] = {"enabled": True, "block_m": bm, "block_n": bn, "block_k": bk,
                    "thread_m": tile["TM"], "thread_n": tile["TN"],
                    "warp_tile": {"warp_m": tile["WM"], "warp_n": tile["WN"]}}
    ir["mapping"] = {"threads_per_block": tile["threads_per_block"], "warps_per_block": tile["warps_per_block"]}
    ir["memory"] = {"use_shared_memory": True, "shared_memory_allocation": "static",
                    "shared_A": {"enabled": True, "shape": [bk, bm+padding] if transpose_a else [bm,bk+padding]},
                    "shared_B": {"enabled": True, "shape": [bk,bn+padding]}}
    ir["pipeline"] = {"async_copy": async_copy, "stage_count": stages, "double_buffering": stages==2}
    ir["resource"] = {"shared_memory": {"total_bytes": stages*shared_bytes(tile, transpose_a, padding)}}
    ir["specialization"] = {"executor": "builtin_simt_v1", "tile": {k: tile[k] for k in TILE_KEYS},
        "transpose_a": transpose_a, "padding": padding, "unroll": unroll, "split_k": split_k,
        "pipeline_stages": stages, "async_copy": async_copy,
        "workspace_bytes": 4*base["problem"]["M"]*base["problem"]["N"]*split_k if split_k > 1 else 0,
        "workspace_lifetime": "per call; allocation/free and reduction included in benchmark"}
    return ir


def specialize_seed(family, tile):
    """Only alter uniquely declared template launch constants, never code regions."""
    source = family["sources"]["cuda_kernel.cuh"]
    start = source.find("LAUNCH_CONFIG_BEGIN")
    end = source.find("LAUNCH_CONFIG_END", start)
    if start < 0 or end < 0 or "template" not in source:
        return None
    region = source[start:end]
    # Keep offsets into the original source; examples in comments are not declarations.
    masked = re.sub(r"/\*.*?\*/|//[^\n]*", lambda m: " "*len(m[0]), source, flags=re.DOTALL)[start:end]
    replacements = []
    for key in TILE_KEYS:
        pattern = rf"\b(?:static\s+)?(?:const\s+int|constexpr\s+int)\s+{key}\s*=\s*(\d+)\s*;"
        matches = list(re.finditer(pattern, masked))
        if len(matches) != 1:
            return None
        match = matches[0]
        replacements.append((match.start(1),match.end(1),str(tile[key])))
    for begin, finish, value in sorted(replacements, reverse=True):
        region = region[:begin]+value+region[finish:]
    result = source[:start] + region + source[end:]
    from SCOPE.specialization.dynamic_shared import normalize_dynamic_shared
    result, _ = normalize_dynamic_shared(result)
    # This generator-owned assertion records launch metadata, not an algorithmic
    # restriction. Keep other static assertions intact so unsupported tiles fail.
    metadata_assert = r'static_assert\(([^;]+),\s*"(?:(?:SCOPE|SCOPE) launch config and shared-memory IR disagree|Compiler.TemplateSpecialization.ShapeStatic: launch config must match specialized shape)"\);'
    def update_metadata(match):
        terms = match.group(1).split('&&')
        if not all(re.fullmatch(r'\s*(' + '|'.join(TILE_KEYS) + r')\s*==\s*\d+\s*', term) for term in terms):
            return match.group(0)
        return re.sub(r'\b(' + '|'.join(TILE_KEYS) + r')\s*==\s*\d+',
                      lambda m: f'{m.group(1)} == {tile[m.group(1)]}', match.group(0))
    result = re.sub(metadata_assert, update_metadata, result)
    # Only the known generated launch-count assertion is metadata. Keep arbitrary
    # user assertions, divisibility requirements and __launch_bounds__ untouched.
    thread_assert = (r'static_assert\(\s*\(BM\s*\*\s*BN\)\s*/\s*\(WM\s*\*\s*WN\)\s*\*\s*32\s*==\s*\d+,\s*'
                     r'"Compiler.TemplateSpecialization.ShapeStatic: threads_per_block must be \d+"\);')
    threads = tile['BM'] * tile['BN'] // (tile['WM'] * tile['WN']) * 32
    return re.sub(thread_assert, lambda m: re.sub(r'==\s*\d+', f'== {threads}',
                  re.sub(r'must be \d+', f'must be {threads}', m.group(0))), result)


def run_search(ir, library, registry, output, config, verify=verify_build_and_run,
               build_platform="windows", benchmark_runs=5, benchmark_warmup_runs=2):
    families = load_families(registry, ir)
    report = {"mode": "deterministic_shape_search", "rounds": [], "results": [],
              "families": [f["family_id"] for f in families], "fallback_required": True}
    output = Path(output)
    if not families:
        report["reason"] = "no_verified_family_for_hardware_and_semantics"
        write_json(output / "search.json", report)
        return report
    problem = ir["problem"]
    if any(problem.get(k) not in (None, "fp32") for k in ("dtype_A", "dtype_B", "dtype_C", "accum_dtype")) or any(
        problem.get(k) is True for k in ("trans_A", "trans_B")) or any(
        problem.get(k) not in (None, "row_major") for k in ("layout_A", "layout_B", "layout_C")):
        report["reason"] = "executor_requires_fp32_row_major_nn"
        write_json(output / "search.json", report)
        return report
    budget = max(1, int(config.get("candidates_per_round", 12)))
    target = float(config.get("target_ratio", 0.8))
    if not 0 < target or not math.isfinite(target):
        raise ValueError("shape_reuse.target_ratio must be finite and positive")
    pool = make_diverse_tiling_pool(build_joint_tiling_candidates(library, ir), budget)
    candidates, seen = [], set()
    harness = {k: v for k, v in families[0]["sources"].items() if k in ("main.cpp", "kernel.h")}
    if set(harness) != {"main.cpp", "kernel.h"}:
        raise ValueError("Archived family must contain main.cpp and kernel.h")

    def attempt(sources, state, round_index, provenance):
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        folder = output / f"r{round_index}" / f"c{len(seen)}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in sources.items():
            (folder / name).write_text(text, encoding="utf-8")
        try:
            result = verify(state, source_dir=folder, build_dir=folder/"build",
                timeout_seconds=int(config.get("timeout_seconds", 180)), build_platform=build_platform,
                benchmark_runs=benchmark_runs, benchmark_warmup_runs=benchmark_warmup_runs)
        except Exception as exc:
            result = copy.deepcopy(state)
            result["verification"] = {"accepted": False, "error_type": type(exc).__name__, "error": str(exc)}
        record = {"round": round_index, "code_dir": str(folder), "source_phase": provenance,
                  "accepted": passed(result), "gflops": metric(result) if passed(result) else None,
                  "relative_to_cublas": ratio(result) if passed(result) else None,
                  "specialization": state.get("specialization"), "verification": result.get("verification"),
                  "performance": result.get("performance", {})}
        summary = result.setdefault("verification", {}).setdefault("summary", {})
        summary.update({f"{k}_status": result["verification"].get(k, {}).get("status")
                        for k in ("compile", "correctness", "runtime_safety")})
        summary["cuda_error"] = result["verification"].get("runtime_safety", {}).get("cuda_error")
        write_json(folder/"specialization.json", record)
        write_json(folder/"verified_ir.json", result)
        report["results"].append(record)
        if record["accepted"]:
            applied = list(state.get("strategy", {}).get("applied_strategy_ids", []))
            candidates.append({"accepted": True, "strategy_id": f"ShapeReuse.{fingerprint[:16]}",
                "verified_ir": result, "source_snapshot": sources, "candidate_code_dir": str(folder),
                "history": {"applied_strategy_ids": applied, "events": [], "failed_strategy_counts": {}},
                "source_phase": provenance})
        write_json(output/"search.json", report)

    # Unchanged implementations compete using measurements on the NEW shape.
    for family in families[:max(1, int(config.get("max_families", 3)))]:
        seed_ir = copy.deepcopy(family["verified_ir"])
        seed_ir.update(problem=copy.deepcopy(problem), hardware=copy.deepcopy(ir["hardware"]))
        for key in ("verification", "performance", "compiler"):
            seed_ir.pop(key, None)
        attempt(family["sources"], seed_ir, 0, "reused_source")

    for round_index in range(1, min(3, max(1, int(config.get("max_rounds", 3))))+1):
        before = len(report["results"])
        supports_async = bool(ir["hardware"].get("supports_cp_async") or
                              (ir["hardware"].get("gpu") or {}).get("supports_cp_async"))
        options = [(False,0,1,1,1,False)] if round_index == 1 else (
            [(True,0,8,1,1,False), (True,1,8,1,1,False), (False,1,8,1,2,False), (True,0,8,1,2,supports_async)]
            if round_index == 2 else [(True,0,8,s,2,supports_async) for s in config.get("split_k_slices", [2,4,8])])
        selected_pool = pool
        if round_index > 1:
            measured_tiles = [r["specialization"]["tile"] for r in sorted(
                (r for r in report["results"] if r["accepted"] and r.get("specialization")),
                key=lambda r:r["gflops"] or 0, reverse=True)]
            selected_pool = sorted(pool, key=lambda t: next((i for i,x in enumerate(measured_tiles)
                                   if all(t[k] == x[k] for k in TILE_KEYS)), len(measured_tiles)))[:3]
        for (transpose, padding, unroll, split, stages, async_copy), tile in itertools.product(options, selected_pool):
            if len(report["results"])-before >= budget:
                break
            split = int(split)
            if split < 1 or split > math.ceil(problem["K"]/tile["BK"]):
                continue
            if stages*shared_bytes(tile,transpose,padding) > hardware_value(ir,"max_shared_memory_per_block_bytes",49152):
                continue
            state = instance_ir(ir,tile,transpose,padding,unroll,split,stages,async_copy)
            if state["specialization"]["workspace_bytes"] > int(config.get("max_workspace_bytes", 268435456)):
                continue
            if round_index == 1:
                family = families[0]
                specialized = specialize_seed(family, tile)
                if specialized is not None:
                    seed_ir = copy.deepcopy(family["verified_ir"])
                    seed_ir.update(problem=copy.deepcopy(problem), hardware=copy.deepcopy(ir["hardware"]))
                    seed_ir["tiling"] = copy.deepcopy(state["tiling"])
                    seed_ir["mapping"] = copy.deepcopy(state["mapping"])
                    seed_ir["specialization"] = {"executor": "seed_launch_constants", "family_id": family["family_id"],
                        "tile": {k:tile[k] for k in TILE_KEYS}, "resource_estimate": "requires compilation"}
                    old_ids = seed_ir.setdefault("strategy", {}).get("applied_strategy_ids", [])
                    seed_ir["strategy"]["applied_strategy_ids"] = [sid for sid in old_ids if not sid.startswith(
                        ("Tiling.BlockTile.", "Tiling.WarpTile.", "Tiling.ThreadTile."))] + [
                        f"Tiling.BlockTile.{tile['BM']}x{tile['BN']}x{tile['BK']}",
                        f"Tiling.WarpTile.{tile['WM']}x{tile['WN']}", f"Tiling.ThreadTile.{tile['TM']}x{tile['TN']}"]
                    for key in ("verification", "performance", "compiler", "resource"):
                        seed_ir.pop(key, None)
                    attempt({**family["sources"], "cuda_kernel.cuh": specialized}, seed_ir,
                            round_index, "reused_parameterized_source")
                    if len(report["results"])-before >= budget:
                        break
            sources = {"main.cpp": harness["main.cpp"],
                       "kernel.h": '#pragma once\n#include "cuda_kernel.cuh"\n',
                       "cuda_kernel.cuh": render(tile,transpose,padding,unroll,split,stages,async_copy)}
            attempt(sources,state,round_index,"deterministic_simt_family")
        report["rounds"].append({"round": round_index, "tested": len(report["results"])-before})
        if any((r["relative_to_cublas"] or 0) >= target for r in report["results"]):
            report.update(fallback_required=False, reason="target_reached")
            break
    else:
        report["reason"] = "bounded_search_below_target; run_shape_specific_llm_generation"
    candidates.sort(key=lambda c: metric(c["verified_ir"]) or 0, reverse=True)
    report["top_results"] = sorted((r for r in report["results"] if r["accepted"]),
                                   key=lambda r:r["gflops"] or 0, reverse=True)[:3]
    write_json(output/"search.json", report)
    report["candidates"] = candidates
    return report
