"""Tune an existing SCOPE CUDA implementation without LLM calls or replacement kernels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from SCOPE.generate_ir.ir_extraction import build_extracted_ir
from SCOPE.specialization.shape_search import TILE_KEYS, passed, specialize_seed, write_json
from SCOPE.verification.build_run_verifier import verify_build_and_run
from SCOPE.specialization.tile_enumeration import enumerate_tiles
from SCOPE.specialization.padding_variants import padding_variant

ROOT = Path(__file__).resolve().parent


def read_sources(source):
    source = Path(source).resolve()
    folder = source if source.is_dir() else source.parent
    kernel = "cuda_kernel.cuh" if source.is_dir() else source.name
    if not source.exists():
        raise ValueError(f"Source does not exist: {source}")
    sources = {p.name:p.read_text(encoding="utf-8-sig") for p in folder.iterdir()
               if p.is_file() and p.suffix in {".cu", ".cuh", ".h", ".hpp", ".cpp"}}
    if kernel not in sources or "main.cpp" not in sources:
        raise ValueError("Source must contain the selected kernel and main.cpp benchmark harness")
    return folder, kernel, sources


def clear_results(ir):
    result = copy.deepcopy(ir)
    for key in ("verification", "performance", "code_ast", "code_verification", "defect_diagnosis",
                "patch_generation", "patch_to_code_application", "chain_step", "resource"):
        result.pop(key, None)
    # Never turn a parameter sweep into the compiler's separate resource sweep.
    result.setdefault("compiler", {}).pop("resource_feedback", None)
    return result


def tune_sources(sources, kernel, ir, library, output, *, max_candidates=0, rounds=3,
                 benchmark_runs=5, warmup_runs=2, platform="windows", timeout=180,
                 target_ratio=None, verify=verify_build_and_run, candidate_tiles=None,
                 padding_modes=('original',), max_variant_compilations=None,
                 aggregation='mean'):
    tuning_started = time.perf_counter()
    if max_candidates < 0 or rounds < 1:
        raise ValueError("Candidate budget must be nonnegative (0 means unlimited); rounds must be positive")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty to avoid mixing old measurements: {output}")
    output.mkdir(parents=True, exist_ok=True)
    baseline = clear_results(ir)
    report = {"mode":"existing_cuda_only", "shape":{k:ir["problem"][k] for k in ("M","N","K")},
              "kernel":kernel, "max_candidates":max_candidates, "rounds":[], "results":[],
              "source_hash":hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
              "target_ratio":target_ratio, "target_reached":False, "skipped_variants": [],
              "benchmark_runs": benchmark_runs, "warmup_runs": warmup_runs,
              "aggregation": aggregation}
    seen = set()

    def evaluate(content, tile, round_number, padding_mode='original', layout=None):
        candidate_started = time.perf_counter()
        candidate_sources = {**sources, kernel:content}
        fingerprint = hashlib.sha256(content.encode()).hexdigest()
        if fingerprint in seen:
            return False
        seen.add(fingerprint)
        number = len(report["results"])
        folder = output / f"candidate_{number:04d}"
        folder.mkdir()
        for name, text in candidate_sources.items():
            (folder/name).write_text(text,encoding="utf-8")
        state = clear_results(baseline)
        if tile:
            state.setdefault("tiling", {}).update(enabled=True, block_m=tile["BM"], block_n=tile["BN"],
                block_k=tile["BK"], thread_m=tile["TM"], thread_n=tile["TN"],
                warp_tile={"warp_m":tile["WM"],"warp_n":tile["WN"],
                           "warp_m_iter":tile["WMITER"],"warp_n_iter":tile["WNITER"]})
            state.setdefault("mapping", {}).update(threads_per_block=tile["threads_per_block"],
                                                    warps_per_block=tile["warps_per_block"])
        state["specialization"] = {"executor":"existing_source_parameters", "tile":tile,
            "source_hash":report["source_hash"], "resource_estimate":"compiler measured; original structural strategies preserved"}
        state['specialization']['padding_mode'] = padding_mode
        if layout:
            for name in ('shared_A', 'shared_B'):
                state.setdefault('memory', {}).setdefault(name, {}).update(layout[name])
            state.setdefault('resource', {}).setdefault('shared_memory', {})['total_bytes'] = layout['shared_memory_bytes']
            state['specialization']['strategy_override'] = 'padding changed by deterministic terminal tuning'
        print(f"[{report['shape']}] candidate={number} round={round_number} tile={None if tile is None else {k:tile[k] for k in TILE_KEYS}}",flush=True)
        try:
            result = verify(state,source_dir=folder,build_dir=folder/"build",build_platform=platform,
                timeout_seconds=timeout,benchmark_runs=benchmark_runs,benchmark_warmup_runs=warmup_runs)
        except Exception as exc:
            result = state
            result["verification"] = {"accepted":False,"error_type":type(exc).__name__,"error":str(exc)}
        v = result.get("verification", {})
        selected_gflops = aggregate_value(result, "gflops", aggregation) if passed(result) else None
        selected_cublas = aggregate_value(result, "cublas_gflops", aggregation) if passed(result) else None
        record = {"candidate_id":number,"round":round_number,"code_dir":str(folder.resolve()),
            "tile":{k:tile[k] for k in TILE_KEYS} if tile else None,"source_hash":fingerprint,
            "padding_mode": padding_mode, "layout": layout,
            "accepted":passed(result),"compile_status":v.get("compile",{}).get("status","not_run"),
            "correctness_status":v.get("correctness",{}).get("status","not_run"),
            "runtime_safety_status":v.get("runtime_safety",{}).get("status","not_run"),
            "cuda_error":v.get("runtime_safety",{}).get("cuda_error"),
            "gflops":selected_gflops,
            "latency_ms":aggregate_value(result, "latency_ms", aggregation),
            "relative_to_cublas":(
                selected_gflops / selected_cublas
                if selected_gflops and selected_cublas else None
            ),
            "performance":result.get("performance", {}),
            "evaluation_seconds":round(time.perf_counter()-candidate_started,6)}
        write_json(folder/"verified_ir.json",result)
        write_json(folder/"candidate.json",record)
        report["results"].append(record)
        print(f"  compile={record['compile_status']} correct={record['correctness_status']} GFLOPS={record['gflops']}",flush=True)
        write_json(output/"search.json",report)
        return True

    evaluate(sources[kernel],None,0)
    all_tiles = list(enumerate_tiles(library, ir)) if candidate_tiles is None else list(candidate_tiles)
    report['enumeration_scope'] = ('strategy_library_domains_all_legal_iteration_layouts'
                                   if candidate_tiles is None else 'llm_coupled_ranges_all_legal_iteration_layouts')
    report['legal_candidate_count'] = len(all_tiles)
    report['candidate_order'] = 'Block -> Warp -> Thread -> WMITER/WNITER'
    pool = all_tiles if max_candidates == 0 else all_tiles[:max_candidates]
    report['budget_omitted_count'] = len(all_tiles) - len(pool)
    write_json(output / 'candidate_manifest.json', all_tiles)
    tile_pool = pool
    pool = [(tile, mode) for tile in tile_pool for mode in padding_modes]
    report['padding_modes'] = list(padding_modes)
    report['expanded_variant_count'] = len(pool)
    write_json(output / 'variant_manifest.json', [{'tile': t, 'padding_mode': m} for t, m in pool])
    max_candidates = len(pool)
    family = {"sources":{"cuda_kernel.cuh":sources[kernel]}}
    if pool and specialize_seed(family,pool[0][0]) is None:
        report["stop_reason"] = "unsupported_source_parameters: require unique numeric BM/BN/BK/WM/WN/TM/TN/WMITER/WNITER in LAUNCH_CONFIG"
    else:
        per_round = max(1, (max_candidates+rounds-1)//rounds)
        pending = list(pool)
        for round_number in range(1,rounds+1):
            before = len(report["results"])
            attempted = 0
            while pending and attempted < per_round and len(report["results"])-1 < max_candidates:
                if max_variant_compilations is not None and len(report['results']) - 1 >= max_variant_compilations:
                    report['stop_reason'] = 'compilation_budget_exhausted'
                    break
                tile, mode = pending.pop(0)
                attempted += 1
                content = specialize_seed(family,tile)
                if content is not None:
                    content, layout, reason = padding_variant(content, tile, mode)
                    if reason:
                        report['skipped_variants'].append({'tile': tile, 'padding_mode': mode, 'reason': reason})
                    else:
                        evaluate(content,tile,round_number,mode,layout)
            report["rounds"].append({"round":round_number,"tested":len(report["results"])-before})
            if report.get('stop_reason') == 'compilation_budget_exhausted':
                break
            if target_ratio is not None and any((r["relative_to_cublas"] or 0)>=target_ratio for r in report["results"]):
                report.update(target_reached=True,stop_reason="target_reached")
                break
            if not pending:
                report["stop_reason"] = "budget_exhausted" if report['budget_omitted_count'] else "candidate_pool_exhausted"
                break
        report.setdefault("stop_reason","budget_exhausted")
    top = sorted((r for r in report["results"] if r["accepted"] and r["gflops"] is not None),
                 key=lambda r:r["gflops"],reverse=True)[:3]
    for rank, record in enumerate(top,1):
        record["rank"] = rank
        destination = output/"top_3"/f"rank_{rank}"
        destination.mkdir(parents=True,exist_ok=True)
        original = Path(record["code_dir"])
        for name in (*sources.keys(),"verified_ir.json","candidate.json"):
            (destination/name).write_bytes((original/name).read_bytes())
    report["top_results"] = top
    report["tested_candidate_count"] = len(report["results"])
    report["accepted_candidate_count"] = sum(bool(item.get("accepted")) for item in report["results"])
    report["failed_candidate_count"] = report["tested_candidate_count"] - report["accepted_candidate_count"]
    report["compile_attempt_count"] = report["tested_candidate_count"]
    report["elapsed_seconds"] = round(time.perf_counter()-tuning_started,6)
    write_json(output/"top_3_results.json",top)
    lines = [f"# GEMM {report['shape']}", "", f"Stop reason: {report['stop_reason']}", "",
             "| Rank | Tile (BM/BN/BK, WM/WN, TM/TN) | GFLOPS | Latency ms | / cuBLAS |",
             "|---|---|---|---|---|"]
    for record in top:
        tile = record["tile"]
        label = ", ".join(str(tile[k]) for k in ("BM","BN","BK","WM","WN","TM","TN")) if tile else "original source"
        lines.append(f"| {record['rank']} | {label} | {record['gflops']} | {record['latency_ms']} | {record['relative_to_cublas']} |")
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"search.json",report)
    return report


def aggregate_value(ir, field, aggregation):
    """Select the requested ranking statistic while retaining all raw metrics."""
    performance = ir.get("performance", {}) or {}
    key = f"{field}_mean" if aggregation == "mean" else f"{field}_trimmed_mean"
    value = performance.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    fallback = performance.get(field)
    return float(fallback) if isinstance(fallback, (int, float)) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True,help="CUDA directory or a .cuh file, with main.cpp alongside")
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument("--shapes",type=int,nargs="+",help="Square sizes, e.g. 1024 2048")
    shapes.add_argument("--matrix-size",type=int,nargs=3,metavar=("M","N","K"))
    parser.add_argument("--ir",type=Path,help="Optional source IR; current hardware is probed again")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--build-platform",choices=("windows","linux"),default="windows")
    parser.add_argument("--max-candidates",type=int,default=0,help="0: all legal strategy-domain tuples; positive: explicit partial-test budget")
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--benchmark-runs",type=int,default=5)
    parser.add_argument("--warmup-runs",type=int,default=2)
    parser.add_argument(
        "--aggregation", choices=("mean", "trimmed_mean"), default="mean",
        help="Statistic used for ranking/final Top-3; all raw run metrics are still recorded",
    )
    parser.add_argument("--timeout",type=int,default=180)
    parser.add_argument("--target-ratio",type=float,help="Optional early stop relative to cuBLAS")
    args = parser.parse_args()
    sizes = [args.matrix_size] if args.matrix_size else [[n,n,n] for n in (args.shapes or [1024])]
    if any(min(s)<=0 for s in sizes) or args.max_candidates < 0 or min(args.rounds,args.benchmark_runs,args.timeout)<1 or args.warmup_runs<0:
        parser.error("Sizes and budgets must be positive; warmup-runs must be nonnegative")
    if args.target_ratio is not None and not 0<args.target_ratio<float("inf"):
        parser.error("target-ratio must be finite and positive")
    source_dir,kernel,sources = read_sources(args.source)
    output = (args.output or ROOT/"gemm_code"/"shape_tuning"/uuid4().hex[:10]).resolve()
    if output == source_dir or source_dir in output.parents or output in source_dir.parents:
        parser.error("Output must be separate from the source directory")
    template_path = args.ir or ROOT/"data"/"IRs"/"optir.json"
    template = json.loads(template_path.read_text(encoding="utf-8-sig"))
    base = build_extracted_ir(template,"FP32 row-major NN GEMM on GPU, matrix size 1024x1024x1024")
    library = json.loads((ROOT/"data"/"lib"/"strategy_library.json").read_text(encoding="utf-8"))
    reports = []
    for shape in sizes:
        ir = copy.deepcopy(base)
        ir["problem"].update(zip(("M","N","K"),shape))
        result = tune_sources(sources,kernel,ir,library,output/"x".join(map(str,shape)),
            max_candidates=args.max_candidates,rounds=args.rounds,platform=args.build_platform,
            benchmark_runs=args.benchmark_runs,warmup_runs=args.warmup_runs,timeout=args.timeout,
            target_ratio=args.target_ratio, aggregation=args.aggregation)
        reports.append({"shape":shape,"stop_reason":result["stop_reason"],"top_results":result["top_results"]})
    write_json(output/"summary.json",reports)
    print(f"Results: {output / 'summary.json'}",flush=True)


if __name__ == "__main__":
    main()
