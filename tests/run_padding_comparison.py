"""Controlled current/historical kernel comparison; no LLM, no original edits."""
import argparse
import copy
import json
from pathlib import Path
from SCOPE.tune import read_sources, tune_sources
from SCOPE.generate_ir.ir_extraction import build_extracted_ir
from SCOPE.verification.gemm_semantic_checker import extract_launch_config
from SCOPE.specialization.shape_search import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-platform', default='windows', choices=['windows', 'linux'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ir = build_extracted_ir(json.loads((root/'data/IRs/optir.json').read_text(encoding='utf-8')),
                            'FP32 row-major NN GEMM on GPU, matrix size 1024x1024x1024')
    ir['problem'].update(M=1024, N=1024, K=1024)
    reports = []
    for i, source in enumerate(args.sources):
        _, kernel, sources = read_sources(source)
        tile = extract_launch_config(sources[kernel])
        tile['warps_per_block'] = tile['BM'] * tile['BN'] // (tile['WM'] * tile['WN'])
        tile['threads_per_block'] = tile['warps_per_block'] * 32
        result = tune_sources(sources, kernel, copy.deepcopy(ir), {}, args.output / f'seed_{i}',
                              candidate_tiles=[tile], padding_modes=['none', 'a_only', 'ab'],
                              platform=args.build_platform, benchmark_runs=5, warmup_runs=2)
        reports.append({'source': source, 'results': result['results'],
                        'skipped': result['skipped_variants']})
        write_json(args.output / 'comparison.json', reports)


if __name__ == '__main__':
    main()
