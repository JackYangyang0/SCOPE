"""Bounded layout replacements and load-loop trials on a recognized kernel."""
import copy
import re

from SCOPE.verification.load_transform import fixed_load_iterations, vector_shared_b, mask_comments
from SCOPE.specialization.padding_variants import padding_variant
from SCOPE.verification.gemm_semantic_checker import extract_launch_config


TRIALS = ('fixed_loads', 'none', 'a_only', 'none_vector_b', 'a_only_vector_b',
          'none_vector_b_fixed', 'a_only_vector_b_fixed')


def transform(source, name):
    if name not in TRIALS:
        raise ValueError('Unknown load/layout trial')
    if name == 'fixed_loads':
        return fixed_load_iterations(source), {}
    mode = 'a_only' if name.startswith('a_only') else 'none'
    # Inspect masked comments, but apply only the declaration edits to the original
    # text so anchors remain available to later repair/tuning steps.
    clean = mask_comments(source)
    updated, layout, error = padding_variant(clean, extract_launch_config(source), mode)
    if error:
        raise ValueError(error)
    for array_name in ('As', 'Bs'):
        pattern = r'__shared__\s+float\s+' + array_name + r'\[[^;]+;'
        before = re.search(pattern, clean)
        after = re.search(pattern, updated)
        source = source.replace(before[0], after[0], 1)
    updated = source
    if 'vector_b' in name:
        updated = vector_shared_b(updated)
    if name.endswith('_fixed'):
        updated = fixed_load_iterations(updated)
    return updated, layout


def update_trial_ir(ir, name, layout):
    """A layout trial is an explicit alternative, not preservation of old padding."""
    strategy = ir.setdefault('strategy', {})
    superseded = []
    if layout:
        applied = strategy.get('applied_strategy_ids', [])
        superseded = [sid for sid in applied if sid.startswith('Layout.SharedMemory.Padding')]
        strategy['applied_strategy_ids'] = [sid for sid in applied if sid not in superseded]
        for key in ('shared_A', 'shared_B'):
            ir.setdefault('memory', {}).setdefault(key, {}).update(copy.deepcopy(layout[key]))
        ir.setdefault('resource', {}).setdefault('shared_memory', {})['total_bytes'] = layout['shared_memory_bytes']
        # The old realization state describes the seed, not the replacement source.
        ir.pop('strategy_contract_state', None)
    ir['load_layout_trial'] = {'name': name, 'superseded_strategy_ids': superseded,
                              'layout': copy.deepcopy(layout), 'requires_runtime_verification': True,
                              'tile_changed': False}
    return superseded
