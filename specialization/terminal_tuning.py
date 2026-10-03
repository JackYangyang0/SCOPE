"""Run source-preserving parameter search on verified terminal seeds."""
import copy
import json
from pathlib import Path
from uuid import uuid4

from SCOPE.tune import tune_sources, read_sources
from SCOPE.specialization.shape_search import metric, passed, write_json
from SCOPE.specialization.tile_range_planner import plan_tile_ranges, expand_plan


def tune_terminal_candidates(candidates, library, output, config, platform, runs, warmups, client=None):
    output = Path(output) / uuid4().hex[:10]
    output.mkdir(parents=True)
    seeds = sorted((c for c in candidates if c.get('accepted') and passed(c.get('verified_ir', {}))),
                   key=lambda c: metric(c['verified_ir']) or 0, reverse=True)
    results, reports, seen = [], [], set()
    max_seeds = max(1, int(config.get('max_seeds', 3)))
    for seed in seeds:
        directory = str(Path(seed['candidate_code_dir']).resolve())
        if directory in seen:
            continue
        seen.add(directory)
        if len(reports) >= max_seeds:
            break
        folder = output / f'seed_{len(reports) + 1}'
        phase = 'planning'
        try:
            _, kernel, sources = read_sources(directory)
            limit = int(config.get('max_combinations', 256))
            plan = plan_tile_ranges(client, seed['verified_ir'], sources[kernel], limit,
                                    audit_dir=output / f'{folder.name}.planning')
            write_json(output / f'{folder.name}.tile_plan.json', plan)
            tiles = expand_plan(plan, seed['verified_ir'], limit)
            phase = 'tuning'
            report = tune_sources(sources, kernel, seed['verified_ir'], library, folder,
                                  max_candidates=int(config.get('max_candidates', 0)),
                                  platform=platform, benchmark_runs=runs, warmup_runs=warmups,
                                  timeout=int(config.get('timeout_seconds', 180)), candidate_tiles=tiles,
                                  padding_modes=config.get('padding_modes', ['original', 'none', 'a_only', 'ab']))
            for record in report.get('top_results', []):
                path = Path(record['code_dir'])
                ir = json.loads((path / 'verified_ir.json').read_text(encoding='utf-8'))
                if not record.get('accepted') or not passed(ir):
                    continue
                candidate = copy.deepcopy(seed)
                candidate.update(strategy_id=f"{seed['strategy_id']}.tile_{output.name}_{len(results)}",
                                 verified_ir=ir, accepted=True, candidate_code_dir=str(path),
                                 source_phase='terminal_tile_tuning',
                                 source_snapshot={name: (path / name).read_text(encoding='utf-8') for name in sources})
                candidate['performance'] = copy.deepcopy(ir.get('performance', {}))
                candidate['path'] = list(seed.get('path', [])) + [f"TileSpecialization:{record['tile']}"]
                candidate['path'].append('PaddingSpecialization:' + record.get('padding_mode', 'original'))
                candidate['history'] = copy.deepcopy(seed.get('history', {}))
                candidate['history'].setdefault('events', []).append({
                    'stage': 'TerminalTileTuning', 'tile': record['tile'],
                    'seed_code_dir': directory, 'code_dir': str(path)})
                results.append(candidate)
            reports.append({'seed': directory, 'report': str(folder / 'search.json'),
                            'status': 'completed', 'tuning_executed': True,
                            'proposed_candidate_count': len(tiles),
                            'stop_reason': report.get('stop_reason')})
        except Exception as exc:
            reports.append({'seed': directory, 'status': 'failed',
                            'failed_phase': phase, 'tuning_started': phase == 'tuning',
                            'original_candidate_preserved': True,
                            'error_type': type(exc).__name__, 'error': str(exc)})
        write_json(output / 'summary.json', reports)
    summary = {'stage': 'TerminalTileTuning', 'phase': 'terminal_tile_tuning',
               'status': ('completed' if reports and all(r.get('status') == 'completed' for r in reports)
                          else 'partial_failure' if results else 'no_tuning_results'),
               'terminal_chain_count': len(results), 'seed_reports': reports,
               'output_dir': str(output), 'max_candidates': config.get('max_candidates', 0)}
    write_json(output / 'summary.json', summary)
    return {'verified_candidates': results, 'summary': summary}
