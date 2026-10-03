"""Fixed-source one-factor benchmarks. No LLM, no tile search, no source overwrite."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from SCOPE.tune_cuda import read_sources, tune_sources
from SCOPE.generate_ir.ir_extraction import build_extracted_ir
from SCOPE.specialization.kernel_controls import control_variants
from SCOPE.specialization.shape_search import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ir', type=Path, help='Source IR to preserve compiler settings for every control')
    parser.add_argument('--matrix-size', nargs=3, type=int, default=[1024, 1024, 1024])
    parser.add_argument('--build-platform', choices=['windows', 'linux'], default='windows')
    parser.add_argument('--benchmark-runs', type=int, default=5)
    parser.add_argument('--warmup-runs', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--prepare-only', action='store_true', help='Freeze variants without compilation or execution')
    args = parser.parse_args()
    root, kernel, sources = read_sources(args.source)
    output = args.output.resolve()
    if root == output or root in output.parents or output in root.parents:
        parser.error('Output must be separate from source')
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be new or empty')
    if min(args.matrix_size) <= 0 or args.benchmark_runs < 1 or args.warmup_runs < 0:
        parser.error('Invalid shape or measurement count')
    template = json.loads((Path(__file__).parent / 'data/IRs/optir.json').read_text(encoding='utf-8-sig'))
    ir = build_extracted_ir(template, 'FP32 row-major NN GEMM on GPU, matrix size ' + 'x'.join(map(str, args.matrix_size)))
    if args.ir:
        saved_ir = json.loads(args.ir.read_text(encoding='utf-8-sig'))
        ir['compiler'] = copy.deepcopy(saved_ir.get('compiler') or {})
    ir['problem'].update(zip(('M', 'N', 'K'), args.matrix_size))
    variants, skipped = control_variants(sources[kernel])
    report = {'source': str(root), 'shape': args.matrix_size, 'one_factor_only': True,
              'skipped': skipped, 'results': [], 'prepared_only': args.prepare_only}
    write_json(output / 'frozen_sources.json', sources)
    write_json(output / 'frozen_benchmark_ir.json', ir)
    # Repeat the baseline at the end to expose measurement drift.
    variants.append(('baseline_repeat', sources[kernel]))
    for name, code in variants:
        case_sources = {**sources, kernel: code}
        item = {'variant': name, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()}
        if args.prepare_only:
            folder = output / name
            folder.mkdir(parents=True, exist_ok=True)
            for filename, content in case_sources.items():
                (folder / filename).write_text(content, encoding='utf-8')
            item['status'] = 'prepared_not_measured'
            report['results'].append(item)
            continue
        try:
            result = tune_sources(case_sources, kernel, copy.deepcopy(ir), {}, output / name,
                                  candidate_tiles=[], platform=args.build_platform,
                                  benchmark_runs=args.benchmark_runs, warmup_runs=args.warmup_runs,
                                  timeout=args.timeout)
            item['measurement'] = result['results'][0]
        except Exception as exc:
            item.update(status='failed', error=str(exc))
        report['results'].append(item)
        write_json(output / 'comparison.json', report)
    ranked = [r for r in report['results'] if r.get('measurement', {}).get('accepted')
              and r['measurement'].get('gflops') is not None and r['variant'] != 'baseline_repeat']
    report['ranking'] = [r['variant'] for r in sorted(ranked, key=lambda r: r['measurement']['gflops'], reverse=True)]
    baselines = [r.get('measurement', {}) for r in report['results'] if r['variant'] in ('baseline', 'baseline_repeat')]
    report['comparison_valid'] = len(baselines) == 2 and all(b.get('accepted') for b in baselines)
    if args.prepare_only:
        report['conclusion'] = 'Prepared sources only; no compile, correctness or performance claim.'
    elif not report['comparison_valid']:
        report['ranking'] = []
        report['conclusion'] = 'Baseline failed verification; do not infer performance improvement from this run.'
    write_json(output / 'comparison.json', report)
    print('Comparison:', output / 'comparison.json')


if __name__ == '__main__':
    main()
