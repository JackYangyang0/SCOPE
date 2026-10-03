"""Single configuration and entry point for source-preserving shape tuning."""
import copy
import json
import warnings
from pathlib import Path

from SCOPE.specialization.top1_shape_transfer import transfer_top1
from SCOPE.tune import read_sources
from SCOPE.specialization.shape_search import passed


def current_shape_candidates(report, seed, current_ir):
    """Import only measured, passing results for the original target shape."""
    results = []
    if not seed:
        return results
    shape = {k: current_ir['problem'][k] for k in ('M', 'N', 'K')}
    for item in report.get('shapes', []):
        if item.get('shape') != shape:
            continue
        for record in item.get('top_results', []):
            if not record.get('accepted'):
                continue
            folder = Path(record['code_dir'])
            ir = json.loads((folder / 'verified_ir.json').read_text(encoding='utf-8'))
            if not passed(ir) or any(ir['problem'][k] != shape[k] for k in shape):
                continue
            _, _, sources = read_sources(folder)
            candidate = copy.deepcopy(seed)
            candidate.update(strategy_id=str(seed.get('strategy_id', 'seed')) + '.shape_' + folder.parent.parent.parent.name + '_' + folder.name,
                             accepted=True, verified_ir=ir, performance=copy.deepcopy(ir.get('performance', {})),
                             candidate_code_dir=str(folder), source_snapshot=sources,
                             source_phase='shape_specialization')
            candidate['path'] = list(seed.get('path', [])) + ['ShapeTile:' + str(record.get('tile')), 'Padding:' + record.get('padding_mode', 'original')]
            results.append(candidate)
    return results


def resolve_config(config, reuse_mode='auto', reuse_family=None):
    for old in ('shape_reuse', 'top1_shape_transfer', 'terminal_tile_tuning'):
        if old in config:
            warnings.warn(f'{old} is no longer an app tuning entry; configure shape_specialization instead.', stacklevel=2)
    settings = copy.deepcopy(config.get('shape_specialization') or {})
    settings.setdefault('enabled', False)
    settings.setdefault('seed_source', 'current_top1')
    if reuse_family is not None:
        settings.update(enabled=True, seed_source='existing', seed_dir=str(reuse_family))
    if reuse_mode == 'only':
        settings.update(enabled=True, seed_source='existing')
    elif reuse_mode == 'off':
        settings['enabled'] = False
    if settings['seed_source'] not in ('current_top1', 'existing'):
        raise ValueError('shape_specialization.seed_source must be current_top1 or existing')
    if settings['enabled'] and settings['seed_source'] == 'existing' and not settings.get('seed_dir'):
        raise ValueError('Existing mode requires shape_specialization.seed_dir or --reuse-family')
    return settings


def load_existing_seed(directory, current_ir):
    folder, kernel, sources = read_sources(directory)
    path = folder / 'verified_ir.json'
    # Historical measurements are not evidence on the current hardware.
    ir = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else copy.deepcopy(current_ir)
    ir['hardware'] = copy.deepcopy(current_ir.get('hardware', {}))
    return {'candidate_code_dir': str(folder), 'verified_ir': ir,
            'source_snapshot': sources, 'kernel_file': kernel, 'requires_revalidation': True}


def specialize_shapes(seed, library, output, config, platform, runs, warmups, client=None):
    return transfer_top1(seed, library, output, config, platform, runs, warmups, client=client)
