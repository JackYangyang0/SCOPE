"""Runtime acceptance and strategy realization are independent decisions."""


def strategy_application(ir, selected_ids):
    reports = {r['strategy_id']: r for r in ir.get('strategy_realization', {}).get('strategy_reports', [])}
    realized, missing, unresolved = [], [], []
    for sid in dict.fromkeys(selected_ids):
        status = reports.get(sid, {}).get('status', 'unknown')
        if status == 'realized':
            realized.append(sid)
        elif status in ('not_realized', 'degraded'):
            missing.append(sid)
        else:
            unresolved.append(sid)
    missing_preserved = [sid for sid, report in reports.items()
                         if sid not in selected_ids and report.get('status') in ('not_realized', 'degraded')]
    missing.extend(missing_preserved)
    return {'selected_strategy_ids': list(selected_ids), 'realized_strategy_ids': realized,
            'missing_preserved_strategy_ids': missing_preserved,
            'missing_strategy_ids': missing, 'unresolved_strategy_ids': unresolved,
            'status': 'not_realized' if missing else 'unproven' if unresolved else 'realized',
            'dependency_update_allowed': not missing and not unresolved,
            'continuation_allowed': bool(ir.get('verification', {}).get('accepted')) and not missing,
            'repair_required': bool(missing)}


def update_strategy_contract_state(ir):
    """Keep requests separate from evidence; never discard unresolved obligations."""
    previous = ir.get('strategy_contract_state', {})
    selected = list(dict.fromkeys(
        previous.get('selected_strategies', [])
        + ir.get('strategy', {}).get('applied_strategy_ids', [])
        + ir.get('locked_repair_contract', {}).get('strategy_ids', [])))
    reports = {r['strategy_id']: r for r in ir.get('strategy_realization', {}).get('strategy_reports', [])}
    ir['strategy_contract_state'] = {
        'selected_strategies': selected,
        'realized_capabilities': [sid for sid in selected if reports.get(sid, {}).get('status') == 'realized'],
        'pending_obligations': [
            {'strategy_id': sid, 'status': reports.get(sid, {}).get('status', 'unknown'),
             'evidence': reports.get(sid, {}).get('evidence', []),
             'message': reports.get(sid, {}).get('message', 'No implementation evidence available')}
            for sid in selected if reports.get(sid, {}).get('status') != 'realized'],
    }
    return ir['strategy_contract_state']


def unproven_dependencies(ir, requirements):
    from fnmatch import fnmatchcase
    from SCOPE.generate_ir.strategy_index_filter import normalize_graph_entries
    state = ir.get('strategy_contract_state', {})
    selected = state.get('selected_strategies', [])
    realized = state.get('realized_capabilities', [])
    missing = []
    for entry in normalize_graph_entries(requirements):
        patterns = entry.get('strategies', [])
        pending = [p for p in patterns if any(fnmatchcase(s, p) for s in selected)
                   and not any(fnmatchcase(s, p) for s in realized)]
        if entry.get('mode', 'all') == 'any':
            if patterns and len(pending) == len(patterns):
                missing.extend(pending)
        else:
            missing.extend(pending)
    return missing
