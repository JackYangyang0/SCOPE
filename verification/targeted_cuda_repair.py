"""Narrow, strategy-preserving repairs for recognized generated CUDA defects."""
import re

from SCOPE.verification.gemm_semantic_checker import extract_launch_config


def repair_known_cuda_defects(source, ir):
    from SCOPE.verification.initial_shared_sync import repair_initial_shared_sync
    source, sync_changes = repair_initial_shared_sync(source)
    from SCOPE.verification.cooperative_load_repair import repair_cooperative_loads
    source, cooperative_changes = repair_cooperative_loads(source, ir)
    from SCOPE.verification.shared_storage_repair import repair_static_shared_overflow
    source, storage_changes = repair_static_shared_overflow(source, ir)
    cooperative_changes.extend(storage_changes)
    cooperative_changes.extend(sync_changes)
    changes = []
    config = extract_launch_config(source)
    problem = ir.get('problem', {})
    row_major = problem.get('layout_A') == 'row_major' and problem.get('trans_A') is False
    divisible = all(isinstance(problem.get(dim), int) and config.get(tile, 0) > 0
                    and problem[dim] % config[tile] == 0
                    for dim, tile in [('M', 'BM'), ('N', 'BN'), ('K', 'BK')])
    if row_major and divisible and config['BK'] % 4 == 0:
        # Match the entire faulty mapping/load/scatter sequence, not symbol names alone.
        pattern = re.compile(
            r'int local_k = (?P<k>vec_idx / \(BM / 4\)|idx / BM);\s*'
            r'int local_m = (?P<m>\(vec_idx % \(BM / 4\)\) \* 4|idx % BM);\s*'
            r'int gm = tile_m0 \+ local_m;\s*'
            r'int gk = (?P<base>bkIdx \* BK \+ )?local_k;\s*'
            r'float4 tmp = FLOAT4\(A\[gm \* K \+ gk\]\);\s*'
            r'As\[(?P<stage>0|mem_flag)\]\[local_k\]\[local_m\]\s*= tmp.x;\s*'
            r'As\[(?P=stage)\]\[local_k\]\[local_m \+ 1\]\s*= tmp.y;\s*'
            r'As\[(?P=stage)\]\[local_k\]\[local_m \+ 2\]\s*= tmp.z;\s*'
            r'As\[(?P=stage)\]\[local_k\]\[local_m \+ 3\]\s*= tmp.w;')
        shared_km = re.search(r'__shared__\s+float\s+As\[2\]\[BK\]\[BM(?:\s*\+\s*1)?\]', source)
        def fix(match):
            is_vector_index = match['k'].startswith('vec_idx')
            if is_vector_index != match['m'].startswith('(vec_idx'):
                return match[0]
            k = '(vec_idx % (BK / 4)) * 4' if is_vector_index else 'idx % BK'
            m = 'vec_idx / (BK / 4)' if is_vector_index else 'idx / BK'
            stage, base = match['stage'], match['base'] or ''
            changes.append({'id':'A_FLOAT4_SCATTER_AXIS', 'stage':stage})
            return f'''int local_k = {k};
        int local_m = {m};
        int gm = tile_m0 + local_m;
        int gk = {base}local_k;
        float4 tmp = FLOAT4(A[gm * K + gk]);
        As[{stage}][local_k][local_m] = tmp.x;
        As[{stage}][local_k + 1][local_m] = tmp.y;
        As[{stage}][local_k + 2][local_m] = tmp.z;
        As[{stage}][local_k + 3][local_m] = tmp.w;'''
        if shared_km:
            source = pattern.sub(fix, source)

    source, duplicates = remove_known_redeclarations(source, config)
    changes.extend(duplicates)
    source, reduction_changes = remove_duplicate_first_tile(source)
    changes.extend(reduction_changes)
    source, schedule_changes = repair_synchronous_double_buffer_schedule(source)
    changes.extend(schedule_changes)
    source, final_tile_changes = repair_guarded_current_tile_load(source)
    changes.extend(final_tile_changes)
    if 'Epilogue.BetaOne.FastPath' in ir.get('strategy', {}).get('applied_strategy_ids', []):
        pattern = r'C\[c_index\] = alpha \* (results\[[^;]+?) \+ C\[c_index\];'
        def beta_fix(match):
            changes.append({'id':'BETA_ONE_FAST_PATH_DISPATCH'})
            return (f'if (beta == 1.0f) {{ C[c_index] = alpha * {match[1]} + C[c_index]; }}\n'
                    f'                    else {{ C[c_index] = alpha * {match[1]} + beta * C[c_index]; }}')
        # Do not rewrite an already dispatched path on later repair attempts.
        if not re.search(r'if\s*\(\s*beta\s*==', source):
            source = re.sub(pattern, beta_fix, source)
    return source, cooperative_changes + changes


def repair_synchronous_double_buffer_schedule(source):
    """Repair a synchronous load-then-compute loop that reads the old stage."""
    marker = re.search(r'/\*\s*\*?\s*MAIN_LOOP_BEGIN\s*\*/', source)
    if not marker:
        return source, []
    loop = re.search(
        r'for\s*\(\s*int\s+bkIdx\s*=\s*1\s*;\s*bkIdx\s*<\s*k_tiles\s*;\s*\+\+bkIdx\s*\)\s*\{',
        source[marker.end():],
    )
    if not loop:
        return source, []
    loop_start = marker.end() + loop.start()
    body_start = marker.end() + loop.end()
    depth, cursor = 1, body_start
    while depth and cursor < len(source):
        depth += (source[cursor] == '{') - (source[cursor] == '}')
        cursor += 1
    if depth:
        return source, []
    loop_body = source[body_start:cursor - 1]
    prefix = source[marker.end():loop_start]
    suffix = source[cursor:]
    old_decl = re.search(
        r'const\s+int\s+comp_flag\s*=\s*\(\s*bkIdx\s*-\s*1\s*\)\s*&\s*1\s*;',
        loop_body,
    )
    mem_decl = re.search(r'const\s+int\s+mem_flag\s*=\s*bkIdx\s*&\s*1\s*;', loop_body)
    if not old_decl or not mem_decl:
        return source, []
    first_compute = re.search(
        r'\b(?:results|accum|acc)\s*(?:\[[^\]]+\])+\s*(?:\+=|=\s*fmaf)',
        loop_body,
    )
    if not first_compute:
        return source, []
    before_compute = loop_body[:first_compute.start()]
    loads_current = (
        re.search(r'\bAs\s*\[\s*mem_flag\s*\].*=', before_compute, re.DOTALL)
        and re.search(r'\bBs\s*\[\s*mem_flag\s*\].*=', before_compute, re.DOTALL)
        and re.search(r'\bbkIdx\s*\*\s*BK\b', before_compute)
        and '__syncthreads()' in before_compute
    )
    computed_tile_zero = bool(re.search(
        r'\b(?:results|accum|acc)\s*(?:\[[^\]]+\])+\s*(?:\+=|=\s*fmaf)', prefix
    ))
    has_final_drain = bool(re.search(
        r'comp_flag\s*=\s*\(\s*k_tiles\s*-\s*1\s*\)\s*&\s*1', suffix
    ))
    if not (loads_current and computed_tile_zero) or has_final_drain:
        return source, []
    repaired_body = (
        loop_body[:old_decl.start()]
        + 'const int comp_flag = bkIdx & 1;'
        + loop_body[old_decl.end():]
    )
    repaired = source[:body_start] + repaired_body + source[cursor - 1:]
    return repaired, [{
        'id': 'SYNCHRONOUS_DOUBLE_BUFFER_READ_STAGE',
        'reason': 'load-then-compute loop must read the stage populated for the current bkIdx',
    }]


def repair_guarded_current_tile_load(source):
    """Unwrap a redundant guard that prevents the final ping-pong producer.

    This repair deliberately recognizes only the prologue/steady-state/drain
    protocol used by the generated skeleton.  A next-tile prefetch using
    ``(bkIdx + 1) * BK`` is a different protocol and is left untouched.
    """
    if not re.search(r"for\s*\(\s*int\s+bkIdx\s*=\s*1\s*;\s*bkIdx\s*<\s*k_tiles", source):
        return source, []
    if not re.search(r"comp_flag\s*=\s*\(\s*k_tiles\s*-\s*1\s*\)\s*&\s*1", source):
        return source, []
    begin = source.find("NEXT_TILE_LOAD_BEGIN")
    end = source.find("NEXT_TILE_LOAD_END", begin + 1)
    if begin < 0 or end < 0:
        return source, []
    region = source[begin:end]
    if not re.search(r"\bbkIdx\s*\*\s*BK\b", region):
        return source, []
    if re.search(r"\(\s*bkIdx\s*\+\s*1\s*\)\s*\*\s*BK\b", region):
        return source, []
    guard = re.search(
        r"if\s*\(\s*(?:bkIdx\s*\+\s*1\s*<\s*k_tiles|"
        r"bkIdx\s*<\s*k_tiles\s*-\s*1)\s*\)\s*\{",
        region,
    )
    if not guard:
        return source, []
    open_brace = region.find("{", guard.start())
    depth = 0
    close_brace = None
    for index in range(open_brace, len(region)):
        if region[index] == "{":
            depth += 1
        elif region[index] == "}":
            depth -= 1
            if depth == 0:
                close_brace = index
                break
    if close_brace is None:
        return source, []
    body = region[open_brace + 1:close_brace]
    if not re.search(r"\bA\s*\[", body) or not re.search(r"\bB\s*\[", body):
        return source, []
    repaired_region = region[:guard.start()] + body + region[close_brace + 1:]
    repaired = source[:begin] + repaired_region + source[end:]
    return repaired, [{
        "id": "FINAL_K_TILE_LOAD_GUARD_REMOVED",
        "reason": "current-bkIdx producer must execute before the final drain",
    }]


def remove_duplicate_first_tile(source):
    """Remove only an identical tile-zero reduction duplicated by a ping-pong loop."""
    region = re.search(r'/\*\s*\*?\s*MAIN_LOOP_BEGIN\s*\*/', source)
    if not region:
        return source, []
    outer = re.search(r'for\s*\(int bkIdx = 1; bkIdx < k_tiles; \+\+bkIdx\)\s*\{', source[region.end():])
    if not outer:
        return source, []
    outer_start = region.end() + outer.start()
    prefix = source[region.end():outer_start]
    inner_start = region.end() + outer.end()
    schedule = re.match(r'\s*__syncthreads\(\);\s*const int comp_flag = \(bkIdx - 1\) & 1;\s*const int mem_flag = bkIdx & 1;', source[inner_start:])
    if not schedule:
        return source, []
    compute_start = inner_start + schedule.end()
    loop = re.match(r'\s*#pragma unroll\s*for \(int k = 0; k < BK; \+\+k\)\s*\{', source[compute_start:])
    if not loop:
        return source, []
    depth, end = 1, compute_start + loop.end()
    while depth and end < len(source):
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    body = source[compute_start:end].replace('[comp_flag]', '[0]')
    if re.sub(r'\s+', '', prefix) != re.sub(r'\s+', '', body) or 'results[' not in body:
        return source, []
    return source[:region.end()] + '\n    ' + source[outer_start:], [{'id':'DUPLICATE_FIRST_K_TILE_REDUCTION'}]


def remove_known_redeclarations(source, config):
    # Scope tracking skips comments and strings. Only side-effect-free, known
    # launch declarations with equal initializers are eligible for removal.
    tokens = re.compile(
        r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|'
        r'(?P<decl>const\s+int\s+(?:thread_num|tid|wid|lane|tile_m0|tile_n0)\s*=\s*[^;]+;|'
        r'dim3\s+(?:threadsPerBlock|blocksPerGrid)\s*\([^;]+;)|[{}]')
    scopes = [{}]
    edits, changes = [], []
    for match in tokens.finditer(source):
        token = match[0]
        if token == '{':
            scopes.append({})
        elif token == '}':
            if len(scopes) > 1:
                scopes.pop()
        elif match['decl']:
            name = re.search(r'\b(thread_num|tid|wid|lane|tile_m0|tile_n0|threadsPerBlock|blocksPerGrid)\b', token)[1]
            normalized = re.sub(r'\s+', '', token)
            identifiers = set(re.findall(r'\b[A-Za-z_]\w*\b', token))
            if not identifiers <= {'const','int','dim3',name,'BM','BN','WM','WN','N','M','CEIL_DIV',
                                   'tid','threadIdx','blockIdx','blockDim','x','y','z'}:
                continue
            previous = scopes[-1].get(name)
            same = previous == normalized
            if name == 'thread_num' and previous:
                alternatives = {'constintthread_num=BM*BN/WM/WN*32;',
                                'constintthread_num=(BM*BN)/(WM*WN)*32;'}
                tiled = all(config.get(k, 0) > 0 for k in ('BM','BN','WM','WN'))
                same |= (tiled and config['BM'] % config['WM'] == 0 and config['BN'] % config['WN'] == 0
                         and previous in alternatives and normalized in alternatives)
            if previous and same:
                edits.append((match.start(), match.end()))
                changes.append({'id':'DUPLICATE_LAUNCH_DECLARATION', 'symbol':name})
            else:
                scopes[-1][name] = normalized
    for start, end in reversed(edits):
        source = source[:start] + source[end:]
    return source, changes
