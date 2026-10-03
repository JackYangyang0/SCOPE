"""Narrow one-factor experiments for the standard fragment GEMM kernel."""
import re
from SCOPE.specialization.padding_variants import padding_variant
from SCOPE.verification.gemm_semantic_checker import extract_launch_config


def protect_pingpong_reuse(source):
    header = 'for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {'
    if source.count(header) != 1:
        raise ValueError('Unsupported pipeline loop')
    start = source.index(header) + len(header)
    depth, end = 1, start
    while depth and end < len(source):
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    if depth:
        raise ValueError('Unbalanced pipeline loop')
    body = source[start:end - 1]
    for required in ('const int comp_flag = (bkIdx - 1) & 1;', 'const int mem_flag = bkIdx & 1;',
                     'As[mem_flag]', 'Bs[mem_flag]', 'As[comp_flag]', 'Bs[comp_flag]'):
        if required not in body:
            raise ValueError('Unrecognized ping-pong ownership')
    active = re.sub(r'/\*.*?\*/|//[^\n]*', '', body, flags=re.S).strip()
    if re.search(r'\b(?:break|continue|return)\b', active):
        raise ValueError('Cannot prove uniform barrier participation')
    if active.endswith('__syncthreads();'):
        return source
    # Uniform K loop: all consumers finish before the next iteration reuses
    # their shared buffer. Existing producer barriers are retained.
    return source[:end - 1] + '\n    __syncthreads(); // Finish consumers before buffer reuse.\n' + source[end - 1:]


def hoist_a_fragment(source):
    pattern = re.compile(
        r'(?P<wn>#pragma unroll\s+for \(int wn = 0; wn < WN / WNITER; \+\+wn\) \{)\s*'
        r'(?P<load>#pragma unroll\s+for \(int i = 0; i < TM; \+\+i\) \{\s*'
        r'regM\[i\] = As\[comp_flag\]\[k(?: \+ u)?\]\[Wrow \* WM \+ wm \* WMITER \+ Trow \* TM \+ i\];\s*\})')
    matches = list(pattern.finditer(source))
    if len(matches) != 2 or len(re.findall(r'regM\[i\]\s*=', source)) != 2:
        raise ValueError('Expected exactly two standard A-fragment loads (steady state and drain)')
    return pattern.sub(lambda m: m['load'] + '\n' + m['wn'], source)


def store_variant(source, width):
    if width not in (1, 4):
        raise ValueError('Unsupported store width')
    tile = extract_launch_config(source)
    if tile.get('TN', 0) % width or tile.get('TN', 0) <= 0:
        raise ValueError('Thread fragment is not divisible by store width')
    pattern = re.compile(r'(\/\*\s*\* STORE_BEGIN\s*\*\/)(.*?)(\/\*\s*\* STORE_END\s*\*\/)', re.S)
    match = pattern.search(source)
    if not match or len(pattern.findall(source)) != 1:
        raise ValueError('Expected one standard STORE region')
    if not any(form in match[2] for form in ('results[m + wm * TM][n + wn * TN]',
                                            'results[wm * TM + m][wn * TN + n]')):
        raise ValueError('Unsupported accumulator-to-store mapping')
    body = '''
#pragma unroll
for (int wm = 0; wm < WM / WMITER; ++wm) {
    #pragma unroll
    for (int wn = 0; wn < WN / WNITER; ++wn) {
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const int gm = tile_m0 + Wrow * WM + wm * WMITER + Trow * TM + m;
            #pragma unroll
            for (int n = 0; n < TN; n += WIDTH) {
                const int gn = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN + n;
BODY
            }
        }
    }
}
'''.replace('WIDTH', str(width))
    scalar = '''if (gm < M && gn < N) {
                    const float value = alpha * results[m + wm * TM][n + wn * TN];
                    C[OFFSET(gm, gn, N)] = beta == 0.0f ? value : value + beta * C[OFFSET(gm, gn, N)];
                }'''
    if width == 4:
        scalar = '''if (gm < M && gn + 3 < N && (reinterpret_cast<unsigned long long>(&C[OFFSET(gm, gn, N)]) & 15ULL) == 0) {
                    float4 out;
''' + '\n'.join(f'                    out.{c} = alpha * results[m + wm * TM][n + {i} + wn * TN];' for i, c in enumerate('xyzw')) + '''
                    if (beta != 0.0f) {
                        const float4 previous = *reinterpret_cast<const float4*>(&C[OFFSET(gm, gn, N)]);
''' + '\n'.join(f'                        out.{c} += beta * previous.{c};' for c in 'xyzw') + '''
                    }
                    *reinterpret_cast<float4*>(&C[OFFSET(gm, gn, N)]) = out;
                } else {
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        if (gm < M && gn + u < N) {
                            const float value = alpha * results[m + wm * TM][n + u + wn * TN];
                            C[OFFSET(gm, gn + u, N)] = beta == 0.0f ? value : value + beta * C[OFFSET(gm, gn + u, N)];
                        }
                    }
                }'''
    return source[:match.start(2)] + body.replace('BODY', scalar) + source[match.end(2):]


def control_variants(source):
    tile = extract_launch_config(source)
    variants = [('baseline', source)]
    skipped = []
    for mode in ('none', 'a_only', 'ab'):
        code, _, reason = padding_variant(source, tile, mode)
        if reason:
            skipped.append({'variant': 'padding_' + mode, 'reason': reason})
        elif code == source:
            skipped.append({'variant': 'padding_' + mode, 'reason': 'identical to baseline'})
        else:
            variants.append(('padding_' + mode, code))
    for label, transform in [('hoist_a_fragment', hoist_a_fragment),
                             ('store_scalar', lambda s: store_variant(s, 1)),
                             ('store_float4', lambda s: store_variant(s, 4))]:
        try:
            variants.append((label, transform(source)))
        except ValueError as exc:
            skipped.append({'variant': label, 'reason': str(exc)})
    try:
        protected = protect_pingpong_reuse(source)
        if protected != source:
            variants.append(('pipeline_reuse_barrier', protected))
    except ValueError as exc:
        skipped.append({'variant': 'pipeline_reuse_barrier', 'reason': str(exc)})
    skipped.append({'variant': 'pipeline_reorder', 'reason': 'No proven general synchronization transform; original barriers preserved'})
    return variants, skipped
