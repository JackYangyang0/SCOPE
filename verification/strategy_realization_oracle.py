from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from SCOPE.verification.optimization_preservation import active_source, vector_load_widths


CRITICAL_PREFIXES = (
    "Layout.SharedMemory.",
    "Vectorization.GlobalLoad",
    "Epilogue.StoreC.Vectorized",
    "Vectorization.StoreC.",
    "Pipeline.AsyncCopy",
    "Pipeline.CpAsync.",
    "Pipeline.WarpAwareDoubleBuffer",
    "Pipeline.DoubleBuffer",
)


def verify_strategy_realization(
    source_dir: Path,
    strategy_ids: list[str],
    verified_ir: dict[str, Any],
) -> dict[str, Any]:
    """Check whether selected optimizations are observable in generated code."""
    raw_source = _load_kernel_source(source_dir)
    source = active_source(raw_source)
    compiler_text = _compiler_evidence(verified_ir)
    from SCOPE.verification.memory_access_plan import anchor_body
    store = active_source(anchor_body(raw_source, 'STORE'))
    reports = [_check_strategy(strategy_id, source, compiler_text, store) for strategy_id in strategy_ids]
    flags = verified_ir.get('verification', {}).get('compile', {}).get('effective_optimization_flags')
    for index, sid in enumerate(strategy_ids):
        if sid == 'Compiler.FastMath.Enabled':
            status = 'unknown' if flags is None else ('realized' if '--use_fast_math' in flags else 'not_realized')
            reports[index] = _report(sid, True, status, flags or [],
                                     'FastMath requires --use_fast_math in the executed nvcc command; IR intent and source labels are not evidence.')
    hard_failures = [item for item in reports if item["critical"] and item["status"] in {"not_realized", "degraded"}]
    known = [item for item in reports if item["status"] != "unknown"]
    if hard_failures:
        status = "fail"
        realization_status = "not_realized"
    elif known and all(item["status"] == "realized" for item in known):
        status = "pass"
        realization_status = "realized"
    elif known:
        status = "advisory"
        realization_status = "partially_realized"
    else:
        status = "advisory"
        realization_status = "unknown"
    return {
        "status": status,
        "realization_status": realization_status,
        "hard_gate_passed": not hard_failures,
        "critical_failure_strategy_ids": [item["strategy_id"] for item in hard_failures],
        "strategy_reports": reports,
        "evidence_sources": ["kernel_source", "compiler_output"],
    }


def _load_kernel_source(source_dir: Path) -> str:
    parts = []
    for name in ("cuda_kernel.cuh", "kernel_skeleton.cu", "gemm_kernel.cu"):
        path = source_dir / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _compiler_evidence(ir: dict[str, Any]) -> str:
    compile_result = (ir.get("verification", {}) or {}).get("compile", {}) or {}
    return "\n".join(str(compile_result.get(key, "")) for key in ("stdout", "stderr", "command"))


def _check_strategy(strategy_id: str, source: str, compiler_text: str, store_region: str = '') -> dict[str, Any]:
    critical = strategy_id.startswith(CRITICAL_PREFIXES) or "CpAsync" in strategy_id
    tile = re.fullmatch(r'Tiling\.(BlockTile|WarpTile|ThreadTile)\.(\d+(?:x\d+)+)', strategy_id)
    if tile:
        from SCOPE.verification.gemm_semantic_checker import extract_launch_config
        keys = {'BlockTile': ('BM', 'BN', 'BK'), 'WarpTile': ('WM', 'WN'), 'ThreadTile': ('TM', 'TN')}[tile[1]]
        expected = list(map(int, tile[2].split('x')))
        actual = extract_launch_config(source)
        if len(keys) != len(expected) or any(key not in actual for key in keys):
            return _report(strategy_id, False, 'unknown', [], 'Launch parameters could not be resolved.')
        matched = all(actual[key] == value for key, value in zip(keys, expected))
        return _report(strategy_id, False, 'realized' if matched else 'not_realized',
                       [f'{key}={actual[key]}' for key in keys], 'Numeric launch parameters compared with the selected Tile tuple.')
    if strategy_id.startswith('Layout.SharedMemory.Padding'):
        if not strategy_id.startswith(('Layout.SharedMemory.PaddingA.', 'Layout.SharedMemory.PaddingB.', 'Layout.SharedMemory.PaddingAB.')):
            return _report(strategy_id, critical, 'unknown', [], 'Unsupported padding target; explicit stride evidence required.')
        operands = ('As', 'Bs') if 'PaddingAB.' in strategy_id else (('As',) if 'PaddingA.' in strategy_id else ('Bs',))
        if not strategy_id.endswith('.Plus1'):
            return _report(strategy_id, critical, 'unknown', [], 'Unsupported padding amount; do not infer padding from shared storage.')
        evidence = []
        for operand in operands:
            match = re.search(r'__shared__\s+float\s+' + operand + r'\s*((?:\[[^\]]+\]\s*){2,3});', source)
            if not match:
                return _report(strategy_id, critical, 'unknown', evidence, 'Unrecognized shared representation; explicit stride evidence required.')
            dims = re.findall(r'\[([^\]]+)\]', match[1])
            last = re.sub(r'\s+', '', dims[-1])
            if last in ('BM', 'BN', 'BK'):
                return _report(strategy_id, critical, 'not_realized', [match[0]], 'Selected padding is absent from the shared row stride. Repair declaration and all address calculations together.')
            if not re.fullmatch(r'(?:BM|BN|BK)\+1', last):
                return _report(strategy_id, critical, 'unknown', [match[0]], 'Padding expression requires evaluation; not proven absent.')
            evidence.append(match[0])
        return _report(strategy_id, critical, 'partially_realized', evidence, 'Plus-one row stride observed; producer/consumer and shared vector alignment must also be validated.')
    if strategy_id == 'Vectorization.StoreC.AlignedNoGuard':
        if not store_region:
            return _report(strategy_id, True, 'unknown', [], 'Cannot isolate output region; no-guard property is unproven.')
        guards = re.findall(r'\bif\s*\([^;{}]*(?:<\s*[MN]\b|uintptr_t|&\s*(?:7|15)u?\b)[^;{}]*\)', store_region)
        if guards:
            return _report(strategy_id, True, 'not_realized', guards, 'Output region still has boundary/alignment guards. Do not remove them unless shape, alignment and ownership are proven.')
        return _report(strategy_id, True, 'partially_realized', [], 'No recognized output boundary/alignment guards; validate vector writes and alignment proof separately.')
    if strategy_id == 'Vectorization.StoreC.GuardedVectorStore':
        return _report(strategy_id, True, 'unknown', [], 'A vector write alone does not establish a correctly guarded vector path.')
    if strategy_id.startswith(('Epilogue.StoreC.Vectorized', 'Vectorization.StoreC.')):
        width_match = re.search(r'float([24])', strategy_id, re.IGNORECASE)
        width = width_match[1] if width_match else '[24]'
        # A vector read of C is not evidence of a vector write to C.
        writes = re.findall(rf'FLOAT{width}\s*\(\s*C\[[^;]+?\]\s*\)\s*=(?!=)', source)
        writes += re.findall(rf'\*\s*reinterpret_cast\s*<\s*float{width}\s*\*\s*>\s*\(\s*&\s*C\[[^;]+?\]\s*\)\s*=(?!=)', source)
        if writes:
            return _report(strategy_id, critical, 'realized', writes, 'Direct vector C write observed; ownership, alignment and bounds still require validation.')
        if width == '4':
            smaller = _check_strategy('Epilogue.StoreC.Vectorized.float2', source, compiler_text)
            if smaller['status'] == 'realized':
                report = _report(strategy_id, critical, 'degraded', smaller['evidence'],
                                 'Requested float4 C store, but recognized only float2 stores. Restore four-column ownership or explicitly reselect; never report the downgrade as implementation success.')
                report.update(requested_vector_width=4, observed_vector_width=2)
                return report
        return _report(strategy_id, critical, 'unknown', [], 'No recognized direct vector C write; aliased pointers require dataflow analysis.')
    if strategy_id == "Layout.SharedMemory.TransposeA":
        declarations = re.findall(r"__shared__\s+float\s+(\w+)\s*((?:\[[^\]]+\]\s*){2,3});", source)
        for name, shape in declarations:
            dims = [re.sub(r"\s+", "", d) for d in re.findall(r"\[([^\]]+)\]", shape)]
            if len(dims) >= 2 and dims[-2:] == ["BM", "BK"] and name == "As":
                return _report(strategy_id, True, "not_realized", [name + shape],
                               "A remains row-major [BM][BK]; repair declaration, producers and consumers together.")
            if len(dims) >= 2 and dims[-2:] == ["BK", "BM"] and name == "As":
                return _report(strategy_id, True, "partially_realized", [name + shape],
                               "Transposed A declaration observed; producer/consumer indexing still needs validation.")
        return _report(strategy_id, True, "unknown", [], "Unrecognized A representation; do not infer transpose from __shared__ alone.")
    if "DoubleBuffer" in strategy_id:
        arrays = re.findall(r"__shared__\s+float\s+(\w+)\s*\[\s*2\s*\]", source)
        report = _report(strategy_id, critical, "partially_realized" if len(arrays) >= 2 else "unknown",
                         arrays, "Two shared buffers alone do not prove load/compute overlap.")
        report["overlap_status"] = "unproven"
        return report
    if strategy_id.startswith("Vectorization.GlobalLoad"):
        width_match = re.search(r"float([24])", strategy_id, re.IGNORECASE)
        widths = {int(width_match.group(1))} if width_match else {2, 4}
        operands = ("A", "B") if "GlobalLoadAB" in strategy_id else (("B",) if "GlobalLoadB" in strategy_id else ("A",))
        observed = vector_load_widths(source)
        evidence = [f"{operand}: float{width} read expression" for operand in operands
                    for width in sorted(observed.get(operand, set()) & widths)]
        complete = all(observed.get(operand, set()) & widths for operand in operands)
        return _report(strategy_id, critical, "realized" if complete else "not_realized", evidence,
                       "Direct vector read evidence; alignment, coverage and correctness require separate verification.")
    patterns, description = _patterns_for(strategy_id)
    if not patterns:
        return _report(strategy_id, critical, "unknown", [], "no robust static signature registered")
    evidence = [pattern for pattern in patterns if re.search(pattern, source, re.IGNORECASE | re.MULTILINE)]
    if evidence:
        return _report(strategy_id, critical, "realized", evidence, description)
    if strategy_id.startswith("Register.") and re.search(r"Used\s+\d+\s+registers", compiler_text, re.IGNORECASE):
        return _report(strategy_id, critical, "partially_realized", ["ptxas register usage"], description)
    return _report(strategy_id, critical, "not_realized" if critical else "unknown", [], description)


def _patterns_for(strategy_id: str) -> tuple[list[str], str]:
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        return [r"\b__shared__\b"], "shared-memory declaration"
    if strategy_id.startswith("Vectorization.GlobalLoad"):
        width_match = re.search(r"float([24])", strategy_id, re.IGNORECASE)
        width = width_match.group(1) if width_match else "[24]"
        vector = rf"float{width}"
        operands = ("A", "B") if "GlobalLoadAB" in strategy_id else (("B",) if "GlobalLoadB" in strategy_id else ("A",))
        operand_patterns = [
            rf"(?:\b{vector}\b[^;\n]*(?:=|\()\s*(?:FLOAT{width}\s*\(\s*)?{operand}\s*\[|"
            rf"reinterpret_cast\s*<\s*(?:const\s+)?{vector}\s*\*[^;\n]*{operand}\s*\[|"
            rf"FLOAT{width}\s*\(\s*{operand}\s*\[)"
            for operand in operands
        ]
        if len(operand_patterns) == 2:
            return [rf"(?s)(?=.*{operand_patterns[0]})(?=.*{operand_patterns[1]})"], "vectorized A/B global loads"
        return operand_patterns, f"vectorized {operands[0]} global load"
    if strategy_id.startswith("Epilogue.StoreC.Vectorized"):
        width_match = re.search(r"float([24])", strategy_id, re.IGNORECASE)
        width = width_match.group(1) if width_match else "[24]"
        return [
            rf"reinterpret_cast\s*<\s*float{width}\s*\*[^;\n]*C\s*\[",
            rf"FLOAT{width}\s*\(\s*C\s*\[",
        ], "vectorized C store"
    if strategy_id.startswith("Pipeline.AsyncCopy") or "CpAsync" in strategy_id:
        return [r"cp\.async", r"cuda::memcpy_async", r"__pipeline_memcpy_async"], "asynchronous copy instruction/API"
    if "DoubleBuffer" in strategy_id:
        return [r"\[[ ]*2[ ]*\].*shared", r"shared.*\[[ ]*2[ ]*\]", r"stage\s*\^\s*1"], "two-stage shared-memory buffer"
    if strategy_id.startswith("Reordering.KLoop.Unroll"):
        return [r"#pragma\s+unroll", r"for\s*\([^;]+;[^;]+;[^\)]+\+=\s*(?:2|4|8|16)"], "unrolled K loop"
    if strategy_id.startswith("Register.AccumulatorLayout"):
        return [r"(?:acc|accum|threadResults?)\s*\["], "register accumulator array"
    return [], ""


def _report(strategy_id: str, critical: bool, status: str, evidence: list[str], message: str) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "critical": critical,
        "status": status,
        "evidence": evidence,
        "message": message,
    }
