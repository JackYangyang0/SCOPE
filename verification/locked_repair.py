"""Build immutable repair context and check explicit parameter changes."""
import difflib
import re
import copy

from SCOPE.verification.gemm_semantic_checker import extract_launch_config
from SCOPE.verification.optimization_preservation import active_source


def begin_optimization_transaction(ir):
    """Retire a previous repair lock only at an explicit new optimization boundary."""
    result = copy.deepcopy(ir)
    previous = result.pop('locked_repair_contract', None)
    verification = result.pop('locked_repair_verification', None)
    if previous is not None:
        result.setdefault('repair_lock_history', []).append({
            'contract': previous, 'verification': verification,
            'retirement_reason': 'new_explicit_optimization_transaction',
        })
    return result


def locked_tile_parameters(ir, source):
    values = extract_launch_config(active_source(source))
    for sid in ir.get("strategy", {}).get("applied_strategy_ids", []):
        for label, keys in (("BlockTile", ("BM", "BN", "BK")),
                            ("WarpTile", ("WM", "WN")), ("ThreadTile", ("TM", "TN"))):
            match = re.fullmatch(rf"Tiling\.{label}\.(\d+(?:x\d+)+)", sid)
            if match:
                numbers = list(map(int, match.group(1).split("x")))
                if len(numbers) == len(keys):
                    values.update(zip(keys, numbers))
    return values


def tile_lock_defects(source, locked):
    actual = extract_launch_config(active_source(source))
    return [f"Locked tile {name} must remain {value}; got {actual.get(name, 'unresolved')}"
            for name, value in locked.items() if actual.get(name) != value]


def repair_evidence(ir, diagnosis, current, baseline):
    from SCOPE.verification.pipeline_evidence import pipeline_evidence, PIPELINE_OBLIGATIONS
    from SCOPE.verification.cooperative_load_repair import diagnose_cooperative_loads
    history = ir.get("strategy", {}).get("history", [])
    compile_error = ir.get('verification', {}).get('compile', {}).get('error_message') or ''
    previous = ir.get("repair_feedback") or {}
    verification = ir.get("verification", {})
    return {
        "defect_diagnosis": compact_diagnosis(diagnosis),
        "strategy_contract_state": ir.get('strategy_contract_state', {}),
        "verification": {
            "summary": verification.get("summary", {}),
            "compile_error": compile_error[:4000],
        },
        "compiler_primary_errors": [line for line in compile_error.splitlines()
                                    if 'error' in line.lower()][:40],
        "source_repair_hints": source_repair_hints(current),
        "pipeline_realization": pipeline_evidence(current),
        "pipeline_repair_obligations": PIPELINE_OBLIGATIONS,
        "cooperative_load_diagnostics": diagnose_cooperative_loads(current, ir),
        "cooperative_load_reference": cooperative_load_reference(current, ir),
        "vector_fast_path_coverage_hints": vector_coverage_hints(current, ir),
        "vector_store_contract": vector_store_contract(current, ir),
        "coupled_repair_obligations": [
            "Resolve declarations and references across the complete function, including outside anchors.",
            "For row-major A float4 loads, consecutive components are K positions, not M positions. "
            "Check thread-to-vector mapping, 16-byte alignment, shared transpose scatter, initial AND next loads together.",
            "Do not assign shared arrays to arrays. Switch buffer indices/pointers with matching producer/consumer barriers.",
            "Preserve exactly-once K coverage, final-tile compute and alpha/beta semantics.",
        ],
        "memory_access_diagnostics": ir.get("memory_access_plan_verification", {}),
        "previous_repair_feedback": {
            key: previous.get(key)
            for key in ("status", "error_message", "failure_class", "method", "targeted_changes")
            if previous.get(key) is not None
        },
        "stage_localization": {
            "status": "hypothesis_only",
            "reported_stage": diagnosis.get("stage"),
            "related_strategy": diagnosis.get("related_strategy"),
            "history": history[-20:] if isinstance(history, list) else history,
        },
        "baseline_status": "repair_baseline_not_assumed_verified",
        "changes_since_repair_baseline": list(difflib.unified_diff(
            baseline.splitlines(), current.splitlines(),
            fromfile="repair_baseline", tofile="current_failed_cuda", lineterm=""))[:500],
    }


def compact_diagnosis(diagnosis):
    """Keep actionable defect evidence without replaying benchmark transcripts."""
    if not isinstance(diagnosis, dict):
        return {}
    defects = []
    for item in diagnosis.get("defects", [])[:16]:
        if not isinstance(item, dict):
            continue
        defects.append({
            key: item.get(key)
            for key in ("defect_type", "related_strategy", "related_fields", "repair_action", "message")
            if item.get(key) is not None
        })
    return {
        key: diagnosis.get(key)
        for key in ("stage", "status", "related_strategy", "defect_count")
        if diagnosis.get(key) is not None
    } | {"defects": defects}


def source_repair_hints(source):
    """Conservative localization hints, not a replacement for runtime validation."""
    code = active_source(source)
    hints = []
    shared = set(re.findall(r"__shared__\s+float\s+(\w+)\s*\[[^\]]+\]\s*\[", code))
    for match in re.finditer(r"\b(\w+)\s*\[[^\]]+\]\s*=\s*(\w+)\s*\[[^\]]+\]\s*;", code):
        if match[1] in shared and match[2] in shared:
            hints.append({"defect_type": "SharedMemory.ArrayAssignment",
                          "evidence": match[0], "repair_action": "Use buffer indices or correctly typed pointers; arrays cannot be assigned."})
    transposed = set(re.findall(r"__shared__\s+float\s+(\w+)\s*\[2\]\s*\[BK\]\s*\[BM(?:\s*\+\s*\w+)?\]", code))
    for load in re.finditer(r"float4\s+(\w+)\s*=\s*FLOAT4\(A\[[^;]+?\]\)\s*;", code):
        # Inspect the immediately following component stores; no variable names
        # other than the A operand and established BK/BM layout are assumed.
        stores = re.match(
            r"\s*(\w+)\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\s*=\s*" + re.escape(load[1]) + r"\.x;"
            r"\s*\1\[\2\]\[\3\]\[([^\]]+)\]\s*=\s*" + re.escape(load[1]) + r"\.y;",
            code[load.end():])
        if stores and stores[1] in transposed and stores[4].strip() != stores[5].strip():
            hints.append({"defect_type": "Vectorization.AScatterAxisSuspect",
                          "evidence": load[0] + stores[0],
                          "repair_action": "For row-major non-transposed A, float4 components advance K. Check shared K/M axes and aligned load mapping in first and next tiles.",
                          "confidence": "conditional_on_row_major_non_transposed_A"})
    return hints


def vector_store_contract(source, ir):
    config = locked_tile_parameters(ir, source)
    return {
        'tile_parameters': config,
        'status': 'repair_obligations_not_a_proof',
        'obligations': [
            'A float4 store needs four consecutive columns owned by this thread, not merely four consecutive accumulator slots.',
            'For WarpLaneFragment2D, derive each accumulator column from wn * WNITER + Tcol * TN + n. With TN=2, adjacent wn fragments need not be adjacent output columns; never join them blindly.',
            'Prove accumulator indices are in range, destination is 16-byte aligned and global_n + 3 < N before any float4 C read or write. Use corresponding 8-byte/two-element requirements for float2.',
            'A tail fallback does not fix an out-of-range accumulator read executed before the guard. Form the vector only inside its valid branch.',
            'If locked ownership cannot support float4, use a coupled ownership-preserving gather/staging protocol with correct synchronization, or report the incompatibility. Do not silently drop the selected strategy or change locked tiles.',
            'Preserve alpha/beta exactly once for every output, including scalar tails. Validate correctness and runtime safety after repair.',
        ],
    }


def cooperative_load_reference(source, ir):
    code = active_source(source)
    if 'cp.async' in code:
        return {'status': 'unsupported_async_protocol'}
    config = locked_tile_parameters(ir, source)
    p = ir.get('problem', {})
    if p.get('layout_A') != 'row_major' or p.get('trans_A') is not False:
        return {'status': 'unsupported_layout'}
    transposed = bool(re.search(r'As\[2\]\[BK\]\[BM', code))
    ordinary = bool(re.search(r'As\[2\]\[BM\]\[BK', code))
    if not (transposed or ordinary) or not re.search(r'Bs\[2\]\[BK\]\[BN', code):
        return {'status': 'unsupported_shared_layout'}
    stores_a = '\n'.join(f'As[stage][{"ak + " + str(i) if transposed else "am"}][{"am" if transposed else "ak + " + str(i)}] = av.{c};'
                         for i, c in enumerate('xyzw'))
    return {
        'status': 'reference_only_not_automatically_applied',
        'contract': 'Adapt stage/k_base to BOTH first and subsequent tile phases. Keep compute, selected tiling and synchronization. This reference assumes full divisible tiles and aligned A/B; otherwise add scalar tail guards. It is not a replacement kernel.',
        'a_vector_count': config.get('BM', 0)*config.get('BK', 0)//4,
        'b_vector_count': config.get('BK', 0)*config.get('BN', 0)//4,
        'reference': '''const int threads = blockDim.x * blockDim.y * blockDim.z;
const int linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
for (int v = linear_tid; v < BM * (BK / 4); v += threads) {
    const int am = v / (BK / 4), ak = (v % (BK / 4)) * 4;
    const float4 av = FLOAT4(A[(tile_m0 + am) * K + k_base + ak]);
''' + stores_a + '''
}
for (int v = linear_tid; v < BK * (BN / 4); v += threads) {
    const int bk = v / (BN / 4), bn = (v % (BN / 4)) * 4;
    const float4 bv = FLOAT4(B[(k_base + bk) * N + tile_n0 + bn]);
    Bs[stage][bk][bn] = bv.x;
    Bs[stage][bk][bn+1] = bv.y;
    Bs[stage][bk][bn+2] = bv.z;
    Bs[stage][bk][bn+3] = bv.w;
}'''}


def vector_coverage_hints(source, ir):
    """Upper bounds for straight-line vector fast paths; unsupported loops stay unknown."""
    from SCOPE.verification.memory_access_plan import anchor_body
    config = locked_tile_parameters(ir, source)
    try:
        threads = config['BM']//config['WM']*(config['BN']//config['WN'])*32
        needed = {'A':config['BM']*config['BK'], 'B':config['BK']*config['BN']}
    except (KeyError, ZeroDivisionError):
        return []
    hints = []
    for region in ('GLOBAL_TO_SHARED_LOAD', 'NEXT_TILE_LOAD'):
        body = active_source(anchor_body(source, region))
        loop_spans = []
        unsupported = False
        for loop in re.finditer(r'\bfor\s*\(', body):
            depth, end = 1, loop.end()
            while end < len(body) and depth:
                depth += (body[end] == '(') - (body[end] == ')')
                end += 1
            start = end
            while start < len(body) and body[start].isspace():
                start += 1
            if start >= len(body) or body[start] != '{':
                unsupported = True
                break
            depth, end = 1, start+1
            while end < len(body) and depth:
                depth += (body[end] == '{') - (body[end] == '}')
                end += 1
            loop_spans.append((start,end))
        if unsupported:
            continue
        for operand in ('A','B'):
            loads = list(re.finditer(r'FLOAT4\(\s*'+operand+r'\[', body))
            if not loads or any(a < load.start() < b for a,b in loop_spans for load in loads):
                continue
            upper = len(loads)*threads*4
            if upper < needed[operand]:
                hints.append({'region':region, 'operand':operand,
                    'required_tile_elements':needed[operand], 'vector_fast_path_upper_bound':upper,
                    'message':'Straight-line float4 fast path cannot fill this tile alone. Add a strided vector-group loop in BOTH load phases, or demonstrate independent scalar coverage. This is a localization hint, not proof of complete dataflow.'})
    return hints
