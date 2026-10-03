"""Small explicit tile controls; derive load transformations after retiling."""
import argparse
import copy
import json
import re
from pathlib import Path

from SCOPE.compare_load_layout import sanitizer_checks
from SCOPE.generate_ir.ir_extraction import build_extracted_ir
from SCOPE.specialization.shape_search import specialize_seed, write_json
from SCOPE.tune_cuda import read_sources, tune_sources
from SCOPE.verification.gemm_semantic_checker import extract_launch_config
from SCOPE.verification.initial_shared_sync import repair_initial_shared_sync
from SCOPE.verification.load_layout_trials import TRIALS, transform


def retile(source, shape, block_tile):
    if len(block_tile) != 3 or len(shape) != 3:
        raise ValueError('Expected three block and problem dimensions')
    if 'scope_fixed_load_offset' in source:
        raise ValueError('Retile the unfixed source first; numeric task counts must be regenerated')
    tile = extract_launch_config(source)
    tile.update(zip(('BM', 'BN', 'BK'), block_tile))
    bm, bn, bk, wm, wn, tm, tn, mi, ni = (tile[k] for k in ('BM', 'BN', 'BK', 'WM', 'WN', 'TM', 'TN', 'WMITER', 'WNITER'))
    if min(bm, bn, bk, wm, wn, tm, tn, mi, ni) <= 0 or any(s % t for s, t in zip(shape, (bm, bn, bk))):
        raise ValueError('Requires positive whole-tile shape')
    if bm % wm or bn % wn or wm % mi or wn % ni or mi % tm or ni % tn or (mi // tm) * (ni // tn) != 32:
        raise ValueError('Invalid warp/thread fragment coverage')
    threads = (bm // wm) * (bn // wn) * 32
    if threads > 1024:
        raise ValueError('Thread limit exceeded')
    tile.update(threads_per_block=threads, warps_per_block=threads // 32)
    updated = specialize_seed({'sources': {'cuda_kernel.cuh': source}}, tile)
    if updated is None:
        raise ValueError('Unsupported launch-parameter region')
    updated = re.sub(r'__launch_bounds__\(\s*\d+\s*,\s*([12])\s*\)',
                     lambda m: f'__launch_bounds__({threads}, {m[1]})', updated)
    return updated, tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tiles', nargs='+', default=['64x64x16', '64x128x16', '128x64x16'])
    parser.add_argument('--load-layout', choices=TRIALS, default='a_only_vector_b')
    parser.add_argument('--matrix-size', nargs=3, type=int, default=[512, 512, 512])
    parser.add_argument('--build-platform', choices=['windows', 'linux'], default='windows')
    parser.add_argument('--sanitizer', action='store_true')
    args = parser.parse_args()
    root, kernel, sources = read_sources(args.source)
    output = args.output.resolve()
    if output.exists() or output == root or root in output.parents or output in root.parents:
        parser.error('Use a new independent output directory')
    source, _ = repair_initial_shared_sync(sources[kernel])
    template = json.loads((Path(__file__).parent / 'data/IRs/optir.json').read_text(encoding='utf-8-sig'))
    ir = build_extracted_ir(template, 'FP32 row-major NN GEMM on GPU, matrix size ' + 'x'.join(map(str, args.matrix_size)))
    ir['problem'].update(zip(('M', 'N', 'K'), args.matrix_size))
    write_json(output / 'frozen_sources.json', sources)
    report = {'source': str(root), 'shape': args.matrix_size, 'load_layout': args.load_layout, 'results': [], 'skipped': []}
    specs = [('control', None)] + [(name, name) for name in args.tiles] + [('control_repeat', None)]
    for name, spec in specs:
        try:
            code, tile = (source, extract_launch_config(source)) if spec is None else retile(source, args.matrix_size, [int(x) for x in spec.split('x')])
            code, _ = transform(code, args.load_layout)
        except (KeyError, ValueError) as exc:
            report['skipped'].append({'variant': name, 'reason': str(exc)})
            continue
        result = tune_sources({**sources, kernel: code}, kernel, copy.deepcopy(ir), {}, output / name,
                              candidate_tiles=[], platform=args.build_platform,
                              benchmark_runs=5, warmup_runs=2, timeout=180)
        item = {'variant': name, 'tile': tile, 'measurement': result['results'][0]}
        if args.sanitizer and item['measurement']['accepted']:
            exe = Path(item['measurement']['code_dir']) / 'build' / ('gemm.exe' if args.build_platform == 'windows' else 'gemm')
            item['sanitizer'] = sanitizer_checks(exe, args.matrix_size, 180)
        report['results'].append(item)
        write_json(output / 'comparison.json', report)
    print('Comparison:', output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
