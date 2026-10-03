"""Resume existing generated terminal chains through the production repair path."""
import shutil
from datetime import datetime
from pathlib import Path


def resume_terminal(args):
    from SCOPE import app
    from SCOPE.llm.openai_client import OpenAICompatibleClient
    from SCOPE.utils.common_utils import load_config, load_json, save_json
    config = load_config(Path(getattr(args, "config", None) or app.DEFAULT_CONFIG))
    client = OpenAICompatibleClient(config['llm'])
    library = None
    resumed_backend = None
    archive = app.ROOT/'results'/'resume_terminal'/datetime.now().strftime('%Y%m%d_%H%M%S')
    archive.mkdir(parents=True, exist_ok=False)
    states = []
    for record_path in sorted((app.ROOT/'results/chain').glob('chain_000*.json')):
        record = load_json(record_path)
        code = record.get('chain_code')
        if not code or not record.get('path'):
            continue
        # Use this chain's persisted IR, never a global latest-IR alias.
        ir_path = Path(record['verified_ir_output'])
        ir = load_json(ir_path)
        backend = app.target_backend(ir)
        if resumed_backend is not None and backend != resumed_backend:
            continue
        resumed_backend = backend
        source = terminal_source_dir(app, record, ir)
        if not source.is_dir():
            continue
        if library is None:
            library_path = app.DEFAULT_CPU_STRATEGY_LIBRARY if backend == "cpu" else app.DEFAULT_STRATEGY_LIBRARY
            library = load_json(Path(library_path))
        snapshot = app.snapshot_source_files(source, app.backend_code_files(ir))
        save_json(archive/(source.name+'.json'), {'record':record, 'ir':ir, 'source_snapshot':snapshot})
        states.append({'code_dir':str(source), 'path':record['path'],
                       'path_code':[int(v) for v in code.split('-')],
                       'current_ir':ir, 'source_snapshot':snapshot,
                       'history':{'applied_strategy_ids':record.get('applied_strategy_ids', []),
                                  'failed_strategy_counts':record.get('failed_strategy_counts', {}),
                                  'events':record.get('events', [])}})
    if not states:
        raise RuntimeError('No persisted terminal chains to resume; output directories were not changed.')
    for name in ('top_3_terminal_results.json', 'evolution_summary.json'):
        path = app.ROOT/'results/check'/name
        if path.exists():
            shutil.copy2(path, archive/name)
    build = config.get('build', {})
    build_platform = args.build_platform or build.get('platform') or 'windows'
    result = app.verify_terminal_chains(states, library or {}, build_platform=build_platform,
        benchmark_runs=args.benchmark_runs or build.get('benchmark_runs', 5),
        benchmark_warmup_runs=args.benchmark_warmup_runs if args.benchmark_warmup_runs is not None else build.get('benchmark_warmup_runs', 2),
        client=client)
    save_json(archive/'summary.json', result['summary'])
    save_json(Path(app.DEFAULT_TOP_RESULTS_OUTPUT), {
        'selection_mode':'resumed_terminal_repair', 'source_archive':str(archive),
        'results':result['top_terminal_results']})
    best = result.get('best_candidate')
    evolution_path = Path(app.DEFAULT_EVOLUTION_OUTPUT)
    evolution = load_json(evolution_path) if evolution_path.exists() else {}
    evolution['selection_mode'] = 'resumed_terminal_repair'
    evolution['terminal_chain_count'] = result['summary']['terminal_chain_count']
    evolution['verified_terminal_chain_count'] = result['summary']['verified_terminal_chain_count']
    evolution['accepted_terminal_chain_count'] = result['summary']['accepted_terminal_chain_count']
    evolution['top_3_terminal_results'] = result['top_terminal_results']
    evolution['best_overall_code_dir'] = best.get('candidate_code_dir') if best else None
    evolution['best_overall_strategy_id'] = result['summary']['best_chain_id']
    evolution['best_overall_gflops'] = result['summary']['best_chain_gflops']
    evolution.setdefault('terminal_resume_history', []).append({'archive':str(archive), 'summary':result['summary']})
    save_json(evolution_path, evolution)
    if best:
        ir = best['verified_ir']
        ir.setdefault('strategy', {})['final_selected_strategy_id'] = best['strategy_id']
        ir['strategy']['final_selected_code_dir'] = best['candidate_code_dir']
        final = app.collect_final_code_from_snapshot(best['source_snapshot'], ir, best['history'], [result['summary']])
        save_json(Path(app.DEFAULT_VERIFIED_IR_OUTPUT), ir)
        save_json(Path(app.DEFAULT_FINAL_CODE_OUTPUT), final)
        app.write_final_code_bundle(Path(app.DEFAULT_FINAL_CODE_BUNDLE_OUTPUT), final)
    print('RESUME SUMMARY: '+str(archive/'summary.json'), flush=True)
    if not best:
        raise RuntimeError('Resumed terminal repair produced no accepted code; inspect the saved summary.')


def terminal_source_dir(app, record, ir):
    persisted = record.get("code_dir")
    if persisted:
        return Path(persisted)
    code = record.get("chain_code")
    return app.backend_generated_code_root(ir) / ("chain." + str(code))


def refresh_saved_resume(archive):
    """Synchronize the UI summary with an already completed, verified resume."""
    from SCOPE import app
    from SCOPE.utils.common_utils import load_json, save_json
    archive = Path(archive)
    top = load_json(Path(app.DEFAULT_TOP_RESULTS_OUTPUT))
    if Path(top.get('source_archive', '')).resolve() != archive.resolve():
        raise ValueError('Top results belong to a different run')
    summary = load_json(archive/'summary.json')
    path = Path(app.DEFAULT_EVOLUTION_OUTPUT)
    evolution = load_json(path) if path.exists() else {}
    evolution.update(selection_mode='resumed_terminal_repair',
        terminal_chain_count=summary['terminal_chain_count'],
        verified_terminal_chain_count=summary['verified_terminal_chain_count'],
        accepted_terminal_chain_count=summary['accepted_terminal_chain_count'],
        top_3_terminal_results=top['results'],
        best_overall_strategy_id=summary['best_chain_id'],
        best_overall_gflops=summary['best_chain_gflops'])
    best = next((r for r in top['results'] if r.get('accepted')), None)
    evolution['best_overall_code_dir'] = best['code_dir'] if best else None
    evolution['terminal_resume_output'] = str(archive/'summary.json')
    save_json(path, evolution)
    if best:
        final_path = Path(app.DEFAULT_FINAL_CODE_OUTPUT)
        final = load_json(final_path)
        content = next((f['content'] for f in final.get('files', []) if f['path'] == 'cuda_kernel.cuh'), None)
        if content != (Path(best['code_dir'])/'cuda_kernel.cuh').read_text(encoding='utf-8'):
            raise ValueError('Final code does not match the verified winner; refusing to relabel it')
        final['final_selected_strategy_id'] = best['chain_id']
        final['final_selected_code_dir'] = best['code_dir']
        save_json(final_path, final)
        app.write_final_code_bundle(Path(app.DEFAULT_FINAL_CODE_BUNDLE_OUTPUT), final)
