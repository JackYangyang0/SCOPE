"""Compare current/reference kernels and bounded coupled layout/tile changes."""
import argparse
import copy
import json
import re
from pathlib import Path

from SCOPE.tune_cuda import read_sources, tune_sources
from SCOPE.specialization.shape_search import specialize_seed, write_json
from SCOPE.specialization.padding_variants import padding_variant
from SCOPE.verification.gemm_semantic_checker import extract_launch_config, strip_cpp_comments
from SCOPE.generate_ir.ir_extraction import build_extracted_ir


def vector_shared_b(source):
    if '__shared__ float Bs[2][BK][BN];' not in source:
        raise ValueError('Requires unpadded standard B storage')
    tile = extract_launch_config(source)
    if tile.get('BN', 0) <= 0 or tile['BN'] % 4:
        raise ValueError('B row/stage stride not aligned')
    if 'const int bn = (v % b_vec_per_row) * 4;' not in source:
        raise ValueError('Unsupported B vector ownership')
    pattern = re.compile(r'Bs\[(?P<s>0|mem_flag)\]\[bk\]\[bn\s*\+\s*0\]\s*=\s*bv.x;\s*'
                         r'Bs\[(?P=s)\]\[bk\]\[bn\s*\+\s*1\]\s*=\s*bv.y;\s*'
                         r'Bs\[(?P=s)\]\[bk\]\[bn\s*\+\s*2\]\s*=\s*bv.z;\s*'
                         r'Bs\[(?P=s)\]\[bk\]\[bn\s*\+\s*3\]\s*=\s*bv.w;')
    updated, count = pattern.subn(lambda m: 'FLOAT4(Bs[' + m['s'] + '][bk][bn]) = bv;', source)
    if count != 2:
        raise ValueError('Expected exactly two complete B scatter groups')
    return updated


def build_variants(current, reference):
    tile = extract_launch_config(reference)
    retiled = specialize_seed({'sources': {'cuda_kernel.cuh': current}}, tile)
    if retiled is None:
        raise ValueError('Unsupported launch constants')
    variants = [('current', current), ('reference', reference), ('tile_only', retiled)]
    for label, source in [('current', current), ('retiled', retiled)]:
        config = extract_launch_config(source)
        clean = strip_cpp_comments(source)
        for mode in ('none', 'a_only'):
            code, _, error = padding_variant(clean, config, mode)
            if error:
                raise ValueError(error)
            variants.append((label + '_' + mode, code))
            variants.append((label + '_' + mode + '_vector_b', vector_shared_b(code)))
    variants.append(('current_repeat', current))
    return variants


def constant_load_stride(source):
    if 'dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);' not in source:
        raise ValueError('Launch geometry is not the proven template thread count')
    old = 'const int threads = blockDim.x * blockDim.y * blockDim.z;'
    if source.count(old) != 1:
        raise ValueError('Unsupported runtime thread expression')
    return source.replace(old, 'const int threads = (BM * BN) / (WM * WN) * 32;')


def static_store(source, size):
    tile = extract_launch_config(source)
    if any(not tile.get(k) or size % tile[k] for k in ('BM', 'BN', 'BK')):
        raise ValueError('No-guard trial requires complete tiles for the benchmark shape')
    guard = 'if (global_m < M && global_n < N)'
    if source.count(guard) != 1:
        raise ValueError('Expected one standard scalar C boundary guard')
    return source.replace(guard, 'if (true) /* shape-specialized complete C tile */')


def fixed_load_iterations(source):
    source = constant_load_stride(source)
    tile = extract_launch_config(source)
    threads = (tile['BM'] // tile['WM']) * (tile['BN'] // tile['WN']) * 32
    for count in (tile['BM'] * tile['BK'] // 4, tile['BK'] * tile['BN'] // 4):
        if count % threads:
            raise ValueError('Fixed load iterations require equal complete vector groups per thread')
    pattern = r'for \(int v = linear_tid; v < ([ab]_vec_count); v \+= threads\) \{'
    result, count = re.subn(pattern, lambda m:
        '#pragma unroll\nfor (int offset = 0; offset < ' + m[1] + '; offset += threads) {\n'
        '    const int v = linear_tid + offset;', source)
    if count != 4:
        raise ValueError('Expected first/next A/B load loops')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--build-platform', choices=['windows', 'linux'], default='windows')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--fixed-loads', action='store_true')
    args = parser.parse_args()
    root, kernel, sources = read_sources(args.source)
    reference_root, reference_kernel, reference = read_sources(args.reference)
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new output directory')
    for folder in (root, reference_root):
        if output == folder or folder in output.parents or output in folder.parents:
            parser.error('Output must be separate from both inputs')
    if args.size <= 0:
        parser.error('Size must be positive')
    variants = build_variants(sources[kernel], reference[reference_kernel])
    if args.refine:
        combined = dict(variants)['retiled_none_vector_b']
        variants = [('current', sources[kernel]), ('reference', reference[reference_kernel]),
                    ('combined', combined), ('combined_constant_stride', constant_load_stride(combined)),
                    ('combined_static_store', static_store(combined, args.size)),
                    ('combined_stride_static_store', static_store(constant_load_stride(combined), args.size)),
                    ('reference_repeat', reference[reference_kernel]), ('current_repeat', sources[kernel])]
    if args.fixed_loads:
        combined = dict(build_variants(sources[kernel], reference[reference_kernel]))['retiled_none_vector_b']
        variants = [('current', sources[kernel]), ('reference', reference[reference_kernel]),
                    ('combined_fixed_loads', fixed_load_iterations(combined)),
                    ('combined_fixed_loads_static_store', static_store(fixed_load_iterations(combined), args.size)),
                    ('current_repeat', sources[kernel])]
    template = json.loads((Path(__file__).parent/'data/IRs/optir.json').read_text(encoding='utf-8-sig'))
    ir = build_extracted_ir(template, f'FP32 row-major NN GEMM on GPU, matrix size {args.size}x{args.size}x{args.size}')
    ir['problem'].update(M=args.size, N=args.size, K=args.size)
    ir['compiler'] = {}  # Same O3/architecture/compiler flags for all controls.
    report = {'shape': args.size, 'source': str(root), 'reference': str(reference_root),
              'same_harness': sources['main.cpp'] == reference['main.cpp'], 'results': []}
    write_json(output/'frozen_current.json', sources)
    write_json(output/'frozen_reference.json', reference)
    for name, code in variants:
        result = tune_sources({**sources, kernel: code}, kernel, copy.deepcopy(ir), {}, output/name,
                              candidate_tiles=[], platform=args.build_platform,
                              benchmark_runs=5, warmup_runs=2, timeout=180)
        report['results'].append({'variant': name, **result['results'][0]})
        write_json(output/'comparison.json', report)
    valid = all(r['accepted'] for r in report['results'] if r['variant'] in ('current', 'current_repeat', 'reference'))
    report['comparison_valid'] = valid
    report['ranking'] = sorted([r for r in report['results'] if r['accepted'] and r['variant'] != 'current_repeat'],
                               key=lambda r: r['gflops'], reverse=True) if valid else []
    write_json(output/'comparison.json', report)
    print(output/'comparison.json', flush=True)


if __name__ == '__main__':
    main()
