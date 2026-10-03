"""Isolated fixed-tile load/layout controls, with optional historical comparison."""
import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

from SCOPE.generate_ir.ir_extraction import build_extracted_ir
from SCOPE.specialization.shape_search import write_json
from SCOPE.tune_cuda import read_sources, tune_sources
from SCOPE.verification.build_run_verifier import (
    run_command, result_text, compile_gemm, DEFAULT_VCVARS64, run_gemm_benchmark, update_run_result,
)
from SCOPE.verification.gemm_semantic_checker import extract_launch_config
from SCOPE.verification.initial_shared_sync import repair_initial_shared_sync
from SCOPE.verification.load_layout_trials import TRIALS, transform


def sanitizer_checks(executable, shape, timeout):
    tool = shutil.which('compute-sanitizer')
    if not tool:
        return {'status': 'not_run', 'reason': 'compute-sanitizer unavailable'}
    checks = []
    for name in ('racecheck', 'synccheck'):
        command = [tool, '--tool', name, '--error-exitcode', '86', '--launch-count', '1',
                   '--kernel-name', 'kns=gemm', str(executable), *map(str, shape)]
        result = run_command(command, cwd=executable.parent, timeout_seconds=timeout)
        checks.append({'tool': name, **result})
        text = result_text(result)
        if result.get('timeout'):
            status = 'inconclusive'
        elif re.search(r'ERROR SUMMARY: [1-9]|RACECHECK SUMMARY: [1-9]', text):
            status = 'fail'
        elif result.get('status') != 'pass' or not re.search(r'(?:ERROR|RACECHECK) SUMMARY: 0', text):
            status = 'inconclusive'
        else:
            status = 'pass'
        checks[-1]['check_status'] = status
    status = ('fail' if any(c['check_status'] == 'fail' for c in checks)
              else 'pass' if all(c['check_status'] == 'pass' for c in checks) else 'inconclusive')
    return {'status': status, 'checks': checks,
            'scope': 'First matching GEMM launch on the requested shape only; not an all-input proof.'}


def diagnose_snapshot(sources, ir, output, platform='windows', timeout=180):
    """Inspect an unfixed frozen kernel without admitting it to performance ranking."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in sources.items():
        (output / name).write_text(content, encoding='utf-8')
    build = output / 'build'
    build.mkdir()
    exe = build / ('gemm.exe' if platform == 'windows' else 'gemm')
    compiled = compile_gemm(ir, output, exe, timeout, DEFAULT_VCVARS64, platform)
    report = {'compile': compiled, 'excluded_from_ranking': True}
    if compiled['status'] == 'pass':
        report['sanitizer'] = sanitizer_checks(exe, [ir['problem'][k] for k in ('M', 'N', 'K')], timeout)
    write_json(output / 'diagnostic.json', report)
    return report


def remeasure_existing(cases, output, platform='windows', rounds=2):
    """Reverse measurement order between rounds; never compile or edit a candidate."""
    output = Path(output)
    if output.exists():
        raise ValueError('Remeasurement output already exists')
    report = {'compiled_again': False, 'benchmark_runs': 5, 'warmup_runs': 2, 'results': []}
    for round_number in range(1, rounds + 1):
        order = cases if round_number % 2 else list(reversed(cases))
        for name, folder in order:
            folder = Path(folder).resolve()
            ir = json.loads((folder / 'verified_ir.json').read_text(encoding='utf-8'))
            exe = folder / 'build' / ('gemm.exe' if platform == 'windows' else 'gemm')
            digest = hashlib.sha256(exe.read_bytes()).hexdigest()
            run = run_gemm_benchmark(exe, ir, 180, 5, 2)
            update_run_result(ir, run)
            item = {'round': round_number, 'variant': name, 'code_dir': str(folder),
                    'executable_sha256': digest, 'performance': ir.get('performance', {}),
                    'verification': ir.get('verification', {}), 'run': run,
                    'executable_unchanged': hashlib.sha256(exe.read_bytes()).hexdigest() == digest}
            report['results'].append(item)
            write_json(output, report)
            print('Remeasure:', round_number, name, item['performance'].get('gflops'), flush=True)
    return report


def finalize_report(report, sources):
    def eligible(item):
        return item['measurement']['accepted'] and item.get('sanitizer', {}).get('status') != 'fail'
    controls = [r for r in report['results'] if r['variant'] in ('control', 'control_repeat')]
    report['comparison_valid'] = len(controls) == 2 and all(eligible(c) for c in controls)
    ranked = sorted((r for r in report['results'] if eligible(r) and r['variant'] not in ('control_repeat', 'historical_reference')),
                    key=lambda r: r['measurement']['gflops'], reverse=True)
    report['ranking'] = [r['variant'] for r in ranked] if report['comparison_valid'] else []
    report['all_executed_sanitizer_checks_passed'] = bool(report.get('sanitizer_requested')) and all(
        r.get('sanitizer', {}).get('status') == 'pass' for r in report['results'])
    try:
        report['inputs_unchanged'] = read_sources(Path(report['source']))[2] == sources
    except (ValueError, OSError) as exc:
        # Results are tied to frozen snapshots even if another run removes its seed.
        report['inputs_unchanged'] = False
        report['source_recheck_error'] = str(exc)
    report['measurement_scope'] = 'Frozen source snapshots; not any subsequently regenerated source directory.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--matrix-size', nargs=3, type=int, default=[512, 512, 512])
    parser.add_argument('--build-platform', choices=['windows', 'linux'], default='windows')
    parser.add_argument('--benchmark-runs', type=int, default=5)
    parser.add_argument('--warmup-runs', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--sanitizer', action='store_true')
    args = parser.parse_args()
    root, kernel, sources = read_sources(args.source)
    output = args.output.resolve()
    if min(args.matrix_size) <= 0 or args.benchmark_runs < 1 or args.warmup_runs < 0 or args.timeout <= 0:
        parser.error('Invalid measurement arguments')
    if root == output or root in output.parents or output in root.parents:
        parser.error('Output must be separate from source')
    if output.exists():
        parser.error('Output must be new; existing results are never overwritten')
    template = json.loads((Path(__file__).parent / 'data/IRs/optir.json').read_text(encoding='utf-8-sig'))
    ir = build_extracted_ir(template, 'FP32 row-major NN GEMM on GPU, matrix size ' + 'x'.join(map(str, args.matrix_size)))
    ir['problem'].update(zip(('M', 'N', 'K'), args.matrix_size))
    source, repairs = repair_initial_shared_sync(sources[kernel])
    variants = [('control', source)]
    report = {'source': str(root), 'shape': args.matrix_size, 'tile': extract_launch_config(source),
              'initial_sync_repairs': repairs, 'skipped': [], 'results': [],
              'sanitizer_requested': args.sanitizer,
              'benchmark_runs': args.benchmark_runs, 'warmup_runs': args.warmup_runs}
    write_json(output / 'frozen_sources.json', sources)
    for name in TRIALS:
        try:
            code, _ = transform(source, name)
            variants.append((name, code))
        except (ValueError, KeyError) as exc:
            report['skipped'].append({'variant': name, 'reason': str(exc)})
    if args.reference:
        ref_root, ref_kernel, ref_sources = read_sources(args.reference)
        report['reference_source'] = str(ref_root)
        report['original_harness_identical'] = ref_sources.get('main.cpp') == sources.get('main.cpp')
        write_json(output / 'frozen_reference.json', ref_sources)
        variants.append(('historical_reference', ref_sources[ref_kernel]))
    variants.append(('control_repeat', source))
    for name, code in variants:
        print('Evaluating:', name, flush=True)
        result = tune_sources({**sources, kernel: code}, kernel, copy.deepcopy(ir), {}, output / name,
                              candidate_tiles=[], platform=args.build_platform,
                              benchmark_runs=args.benchmark_runs, warmup_runs=args.warmup_runs,
                              timeout=args.timeout)
        measurement = result['results'][0]
        item = {'variant': name, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
                'tile': extract_launch_config(code), 'measurement': measurement}
        if args.sanitizer and measurement['accepted']:
            exe = Path(measurement['code_dir']) / 'build' / ('gemm.exe' if args.build_platform == 'windows' else 'gemm')
            item['sanitizer'] = sanitizer_checks(exe, args.matrix_size, args.timeout)
        report['results'].append(item)
        write_json(output / 'comparison.json', report)
    finalize_report(report, sources)
    write_json(output / 'comparison.json', report)
    print('Comparison:', output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
