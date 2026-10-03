"""Compact, advisory implementation and measurement context for CUDA prompts.

These summaries guide generation, never decide acceptance or prove performance.
"""
import copy
import hashlib
import math
import re

from SCOPE.verification.optimization_preservation import active_source, staged_buffers, vector_load_widths
from SCOPE.verification.unlock_evidence import classify_diagnosis


IMPLEMENTATION_POLICY = (
    "Strategy labels and source tokens are not evidence of equivalent implementations or speed. "
    "Understand the complete CURRENT parent kernel, including prologue, steady state, drain and "
    "epilogue; return only coupled region edits unless full-kernel locked repair is explicitly requested. Explicit "
    "patch contracts remain binding. Preserve unrelated selected optimizations and FP32 semantics. "
    "In the existing expected_effect/potential_risks or code_generation_notes fields, briefly state "
    "the actual implementation change and its tradeoff, not a promised speedup. During repair, "
    "fix the diagnosed implementation while retaining locked strategies; do not tune by deleting them. "
    "Do not invent profiling results. Static branch/instruction counts, buffer declarations and "
    "vector types do not prove executed instruction costs, overlap or a bottleneck. Performance "
    "must be measured after correctness under the same shape, precision, hardware and harness. "
    "Only omit a bounds/alignment/tail check when the selected fast-path policy permits it AND "
    "current shape, launch, ownership, physical strides and access width prove it redundant; "
    "otherwise retain a valid guard/fallback. Never remove synchronization solely for speed."
)


def is_cpu_request(ir, strategy):
    return (str((ir.get("target") or {}).get("backend", "")).lower() == "cpu"
            or str(strategy.get("strategy_id", "")).startswith("CPU."))


def is_harness_file(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1] in (
        "main.cpp", "main.cu", "main.c", "main.cuh")


def kernel_implementation_summary(source):
    """Recognize supported source forms; a missing match means unknown, not absent."""
    code = active_source(source)
    # Alignment qualifiers do not change the shape of staged array declarations.
    declarations = re.sub(r"__align__\s*\([^)]*\)|alignas\s*\([^)]*\)", "", code)
    buffers = {**staged_buffers(source), **staged_buffers(declarations)}
    widths = vector_load_widths(source)
    shared_stores = []
    for operand in ("As", "Bs"):
        if re.search(rf"\bFLOAT4\s*\(\s*{operand}\s*\[[^;]+?\)\s*=", code):
            shared_stores.append(operand)
        elif re.search(rf"\*\s*reinterpret_cast\s*<\s*float4\s*\*\s*>\s*"
                       rf"\(\s*(?:&\s*{operand}\s*\[|{operand}\s*\+)[^;]+?\)\s*=", code):
            shared_stores.append(operand)
    return {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "basis": "advisory source patterns, not execution or semantic proof; unmatched forms are unknown",
        "global_load_widths_observed": {key: sorted(widths[key]) if key in widths else None
                                        for key in ("A", "B")},
        "shared_float4_store_targets_observed": shared_stores,
        "staged_shared_arrays_observed": buffers,
        "double_buffer_storage_observed": True if any(n >= 2 for n in buffers.values()) else None,
        "async_copy_syntax_observed": True if re.search(r"cp\.async|\bmemcpy_async\b", code) else None,
        "launch_bounds_observed": re.findall(r"__launch_bounds__\s*\(([^)]*)\)", code)[:4],
        "unroll_directives_observed": re.findall(r"(?m)^\s*#pragma\s+unroll\b([^\r\n]*)", code)[:8],
        "overlap_status": "unproven; inspect load issue, independent compute, waits and stage reuse",
        "interpretation": "A global float4 read may require scalar shared scatter. Neither store width "
                          "nor two buffers establishes a faster implementation. Compare generated "
                          "instructions and measured latency; do not rank by static instruction counts.",
    }


def compact_verifier_feedback(ir):
    verification = ir.get("verification") or {}
    summary = verification.get("summary") or {}
    metrics = {**(ir.get("performance") or {}),
               **{k: v for k, v in summary.items() if v is not None}}
    keys = ("latency_ms", "gflops", "latency_ms_mean", "latency_ms_trimmed_mean",
            "latency_ms_median", "latency_ms_std", "gflops_mean", "gflops_trimmed_mean",
            "gflops_std", "benchmark_runs", "benchmark_successful_runs", "warmup_runs",
            "benchmark_warmup_runs", "relative_to_cublas", "cublas_gflops_mean")
    statuses = {key: summary.get(key, (verification.get(node) or {}).get("status"))
                for key, node in (("compile_status", "compile"), ("correctness_status", "correctness"),
                                  ("runtime_safety_status", "runtime_safety"))}
    diagnosis = classify_diagnosis(ir.get("defect_diagnosis") or {}, statuses)
    defects = diagnosis.get("defects") or []
    compiler = (verification.get("compile") or {}).get("ptxas_resources") or {}
    return {
        "scope": "last recorded verification; may be inherited from an earlier version, "
                 "not validation or performance of the proposed patch",
        **statuses,
        "cuda_error": summary.get("cuda_error"),
        "metrics": {k: metrics[k] for k in keys if isinstance(metrics.get(k), (int, float))
                    and not isinstance(metrics[k], bool) and math.isfinite(metrics[k])},
        "compiler_resources": {k: copy.deepcopy(compiler[k]) for k in
                               ("register_counts", "spill_bytes_sum") if k in compiler},
        "compiler_resource_scope": "reported compiled functions, not measured occupancy or proof of a bottleneck",
        "semantic_defect_count": len(defects),
        "semantic_defect_summary": [
            {"defect_type": d.get("defect_type"),
             "kind": d.get("selection_evidence_kind"),
             "message": str((d.get("evidence") or {}).get("message", ""))[:400]}
            for d in defects[:4]],
        "failed_strategy_counts": dict(list((ir.get("strategy", {}).get("failed_strategy_counts") or {}).items())[:12]),
        "changed_regions": (ir.get("strategy", {}).get("changed_regions") or [])[:12],
        "missing_data_policy": "Missing measurements are unknown, not zero or proof of improvement.",
    }


def observed_strategy_implementations(code_summary):
    """Source-bound observations, not strategy acceptance or alignment proofs."""
    observations = {}
    for item in code_summary.get("files", []):
        evidence = item.get("implementation_evidence") or {}
        widths = evidence.get("global_load_widths_observed") or {}
        if not evidence.get("source_sha256") or not all(4 in (widths.get(op) or []) for op in "AB"):
            continue
        observations.setdefault("Vectorization.GlobalLoadAB.float4", []).append({
            "file": item.get("path"), "source_sha256": evidence["source_sha256"],
            "status": "observed_not_proven",
            "basis": "A and B float4 global reads observed in current source",
            "unproven": "coverage of all tiles, alignment, selected contract and performance benefit",
        })
    return observations


def cuda_generation_context(ir, strategy, code_context):
    if is_cpu_request(ir, strategy):
        return code_context
    result = copy.deepcopy(code_context)
    files = result.get("files") or {}
    result["files"] = {path: value for path, value in files.items() if not is_harness_file(path)}
    if isinstance(result.get("patch_anchors"), dict):
        result["patch_anchors"] = {p: a for p, a in result["patch_anchors"].items() if not is_harness_file(p)}
    result["implementation_evidence"] = {
        path: kernel_implementation_summary(value) for path, value in result["files"].items()
        if str(path).endswith((".cu", ".cuh")) and isinstance(value, str)}
    from SCOPE.utils.ablation import model_visible_ir
    result["verifier_feedback"] = compact_verifier_feedback(model_visible_ir(ir))
    return result
