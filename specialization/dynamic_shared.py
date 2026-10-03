"""Recognize the generator-owned two-buffer layout, never arbitrary pointers."""
import re


DECL = r'float\s*\(&(?P<name>As|Bs)\)\[2\]\[BK\]\[(?P<extent>BM|BN)\]\s*=\s*\*reinterpret_cast<float\s*\(\*\)\[2\]\[BK\]\[(?P=extent)\]>\((?P<base>[^;]+)\);'


def normalize_dynamic_shared(source):
    matches = list(re.finditer(DECL, source))
    if len(matches) != 2 or {m['name'] for m in matches} != {'As', 'Bs'}:
        return source, False
    declarations = {m['name']: m for m in matches}
    if declarations['As']['extent'] != 'BM' or declarations['Bs']['extent'] != 'BN':
        return source, False
    if re.sub(r'\s+', '', declarations['As']['base']) != 'scope_shared':
        return source, False
    if re.sub(r'\s+', '', declarations['Bs']['base']) != 'scope_shared+2*BK*BM*sizeof(float)':
        return source, False
    size = '(2 * BK * (BM + BN) * sizeof(float))'
    value = r'(?:\d+|\(2 \* BK \* \(BM \+ BN\) \* sizeof\(float\)\))'
    launch = r'(gemm<[^;]+?>\s*<<<\s*\w+\s*,\s*\w+\s*,)\s*' + value + r'(\s*>>>)'
    attr = r'(cudaFuncAttributeMaxDynamicSharedMemorySize\s*,)\s*' + value + r'(\s*\))'
    if len(re.findall(launch, source)) != 1 or len(re.findall(attr, source)) != 1:
        return source, False
    source = re.sub(launch, lambda m: m[1] + size + m[2], source)
    source = re.sub(attr, lambda m: m[1] + size + m[2], source)
    return source, True


def dynamic_padding(source, tile, mode, static_transform):
    normalized, supported = normalize_dynamic_shared(source)
    if not supported:
        return None, {}, 'unsupported dynamic shared layout or launch'
    # Reuse direct-array validation after removing only the recognized aliases.
    direct = re.sub(DECL, lambda m: f'__shared__ float {m["name"]}[2][BK][{m["extent"]}];', normalized)
    result, metadata, reason = static_transform(direct, tile, mode)
    if reason:
        return result, metadata, reason
    pa = metadata['shared_A']['padding']
    pb = metadata['shared_B']['padding']
    a = 'BM + 1' if pa else 'BM'
    b = 'BN + 1' if pb else 'BN'
    result = re.sub(r'__shared__ float As\[2\]\[BK\]\[[^\]]+\];',
                    f'float (&As)[2][BK][{a}] = *reinterpret_cast<float (*)[2][BK][{a}]>(scope_shared);', result)
    result = re.sub(r'__shared__ float Bs\[2\]\[BK\]\[[^\]]+\];',
                    f'float (&Bs)[2][BK][{b}] = *reinterpret_cast<float (*)[2][BK][{b}]>(scope_shared + 2 * BK * ({a}) * sizeof(float));', result)
    result = result.replace('(2 * BK * (BM + BN) * sizeof(float))',
                            f'(2 * BK * (({a}) + ({b})) * sizeof(float))')
    return result, metadata, None
