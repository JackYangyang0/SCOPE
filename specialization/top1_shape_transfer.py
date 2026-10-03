"""Independently retune a frozen, verified winner for each target shape."""
import copy
from pathlib import Path
from uuid import uuid4

from SCOPE.tune import read_sources, tune_sources
from SCOPE.tune_cuda import clear_results
from SCOPE.specialization.shape_search import passed, write_json, TILE_KEYS
from SCOPE.specialization.tile_range_planner import plan_tile_ranges, expand_plan
from SCOPE.verification.gemm_semantic_checker import extract_launch_config


def diverse_tiles(tiles, limit):
    groups = {}
    for tile in tiles:
        groups.setdefault(tuple(tile[k] for k in ('BM', 'BN', 'BK')), []).append(tile)
    selected = []
    while groups and len(selected) < limit:
        for key in list(groups):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) == limit:
                break
    return selected


def transfer_top1(seed, library, output, config, platform, runs, warmups, client=None):
    output = Path(output).resolve() / uuid4().hex[:10]
    report = {'stage': 'ShapeSpecialization', 'output_dir': str(output), 'shapes': []}
    external = bool(seed and seed.get('requires_revalidation') and config.get('seed_source') == 'existing')
    if not seed or (not external and (not seed.get('accepted') or not passed(seed.get('verified_ir', {})))):
        report['status'] = 'skipped_no_verified_top1'
        write_json(output / 'summary.json', report)
        return report
    if seed.get('source_snapshot'):
        kernel = seed.get('kernel_file', 'cuda_kernel.cuh')
        sources = copy.deepcopy(seed['source_snapshot'])
    else:
        _, kernel, sources = read_sources(seed['candidate_code_dir'])
    report['seed_code_dir'] = seed['candidate_code_dir']
    write_json(output / 'seed_sources.json', sources)
    limit = max(1, int(config.get('max_candidates_per_shape', config.get('max_candidates', 32))))
    expansion_limit = max(limit, int(config.get('max_plan_combinations', 256)))
    shapes = config.get('shapes', [512, 1024, 2048])
    seen = set()
    for size in shapes:
        shape = (size, size, size) if type(size) is int else tuple(size)
        if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('Shape transfer requires positive M/N/K integers')
        if shape in seen:
            continue
        seen.add(shape)
        folder = output / 'x'.join(map(str, shape))
        item = {'shape': dict(zip(('M', 'N', 'K'), shape)), 'output_dir': str(folder)}
        ir = clear_results(seed['verified_ir'])
        ir['problem'].update(item['shape'])
        tiles = []
        try:
            plan = plan_tile_ranges(client, ir, sources[kernel], expansion_limit,
                                    audit_dir=folder / 'planning', allow_bounded=True)
            write_json(folder / 'tile_plan.json', plan)
            tiles = expand_plan(plan, ir, expansion_limit, allow_bounded=True)
            item['expanded_candidate_count'] = len(tiles)
            item['expansion_cap_reached'] = len(tiles) == expansion_limit
            tiles = diverse_tiles(tiles, limit)
            item['selected_tile_count'] = len(tiles)
        except Exception as exc:
            item['planning_error'] = str(exc)
            current = extract_launch_config(sources[kernel])
            if all(type(current.get(k)) is int and current[k] > 0 for k in TILE_KEYS):
                fallback = {'groups': [{
                    'block_tiles': [[current['BM'], current['BN'], bk] for bk in
                                    sorted({current['BK'], max(4, current['BK'] // 2), current['BK'] * 2}) if bk <= 1024],
                    'warp_tiles': [[current['WM'], current['WN']]],
                    'thread_tiles': [[current['TM'], current['TN']]],
                }]}
                tiles = diverse_tiles(expand_plan(fallback, ir, expansion_limit, allow_bounded=True), limit)
                item['fallback'] = 'local_BK_and_iteration_search_preserving_block_warp_thread'
                item['selected_tile_count'] = len(tiles)
                write_json(folder / 'fallback_plan.json', fallback)
        try:
            # Test layout alternatives of the original tile even when planning fails.
            if config.get('padding_modes', ['original', 'none', 'a_only', 'ab']) != ['original']:
                original_tile = extract_launch_config(sources[kernel])
                if all(type(original_tile.get(k)) is int and original_tile[k] > 0 for k in TILE_KEYS):
                    original_tile['warps_per_block'] = original_tile['BM'] * original_tile['BN'] // (original_tile['WM'] * original_tile['WN'])
                    original_tile['threads_per_block'] = original_tile['warps_per_block'] * 32
                    tiles = [original_tile] + tiles
            # Even a failed planner must leave a measured original-tile baseline.
            result = tune_sources(copy.deepcopy(sources), kernel, ir, library, folder / 'tuning',
                                  candidate_tiles=tiles, max_candidates=0, max_variant_compilations=limit,
                                  platform=platform, benchmark_runs=runs, warmup_runs=warmups,
                                  timeout=int(config.get('timeout_seconds', 180)),
                                  padding_modes=config.get('padding_modes', ['original', 'none', 'a_only', 'ab']))
            item.update(status='completed', top_results=result.get('top_results', []))
            item['tested_candidate_count'] = len(result.get('results', []))
            if 'planning_error' in item:
                item['status'] = 'planning_failed_layout_only' if len(result.get('results', [])) > 1 else 'planning_failed_baseline_only'
                if item.get('fallback') and len(result.get('results', [])) > 1:
                    item['status'] = 'fallback_completed'
            if not item['top_results']:
                item['status'] = 'no_verified_result'
        except Exception as exc:
            item.update(status='failed', error=str(exc))
        report['shapes'].append(item)
        write_json(output / 'summary.json', report)
    report['status'] = ('completed' if all(s['status'] == 'completed' for s in report['shapes'])
                        else 'partial_failure')
    write_json(output / 'summary.json', report)
    return report
