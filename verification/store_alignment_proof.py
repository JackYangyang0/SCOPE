"""Conservative proof for the standard row-major fragment store in this harness."""
import re
from SCOPE.verification.optimization_preservation import active_source
from SCOPE.verification.gemm_semantic_checker import extract_launch_config


def prove_fragment_store_alignment(source, harness, ir):
    code = re.sub(r'\s+', '', active_source(source))
    harness = re.sub(r'\s+', '', active_source(harness))
    tile = extract_launch_config(source)
    n = ir.get('problem', {}).get('N')
    if type(n) is not int or n <= 0 or n % 4:
        return None
    if any(type(tile.get(k)) is not int or tile[k] % 4 for k in ('BN', 'WN', 'WNITER', 'TN')):
        return None
    # The proof is scoped to the benchmark call, not arbitrary caller pointers.
    if not re.search(r'cudaMalloc\(&d_C,', harness) or not re.search(r'cuda_gemm\([^;]*,d_C\);', harness):
        return None
    if re.search(r'\bC\s*(?:\+=|-=|=(?!=))', active_source(source)):
        return None
    macros = re.sub(r'\s+', '', re.sub(r'/\*[\s\S]*?\*/|//[^\n]*', '', source))
    if '#defineOFFSET(row,col,ld)((row)*(ld)+(col))' not in macros:
        return None
    if 'constinttile_n0=blockIdx.x*BN;' not in code:
        return None
    columns = re.findall(r'constint(\w+)=tile_n0\+Wcol\*WN\+wn\*WNITER\+Tcol\*TN;', code)
    for name in columns:
        if len(re.findall(r'\b' + re.escape(name) + r'\s*=(?!=)', active_source(source))) != 1:
            return None
        if re.search(r'(?<!\w)' + re.escape(name) + r'(?:\+=|-=|\+\+|--)', code):
            return None
    accesses = list(re.finditer(r'FLOAT4\(C\[(\w+)\]\)', code))
    direct = list(re.finditer(r'FLOAT4\(C\[OFFSET\(\w+,(\w+),N\)\]\)', code))
    if any(m[1] not in columns for m in direct):
        return None
    if not accesses and not direct:
        return None
    if len(re.findall(r'FLOAT4\(C\[', code)) != len(accesses) + len(direct):
        return None
    for access in accesses:
        name = access.group(1)
        definitions = list(re.finditer(r'constint' + re.escape(name) + r'=OFFSET\(\w+,(\w+),N\);', code[:access.start()]))
        if not definitions or definitions[-1].group(1) not in columns:
            return None
        if re.search(r'(?<!\w)' + re.escape(name) + r'(?:\+=|-=|=(?!=))', code[definitions[-1].end():access.start()]):
            return None
    return {'scope': 'current_shape_and_cudaMalloc_benchmark', 'N': n,
            'column_factors': {k: tile[k] for k in ('BN', 'WN', 'WNITER', 'TN')},
            'basis': '16-byte base; row stride and every column summand divisible by four'}
