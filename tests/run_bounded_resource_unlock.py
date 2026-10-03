"""Resume measured resource unlock from the repaired terminal report."""
from datetime import datetime
from pathlib import Path
from SCOPE.utils.common_utils import load_json, save_json
from SCOPE.verification.bounded_resource_unlock import run_bounded_unlock
from SCOPE.app import build_top_terminal_results


def publish_report(root, output):
    records=load_json(output/'report.json')['records']
    measured=[]
    for record in records:
        if record.get('status') not in ('pass','fail'):
            continue
        folder=Path(record['code_dir'])
        ir=load_json(folder/'verified_ir.json')
        summary=ir['verification']['summary']
        summary.update(oracle_acceptance_status='pass' if record.get('accepted') else 'fail',
                       acceptance_basis='compile_correctness_runtime_oracle',code_verification_status='not_run',
                       code_verification_basis='bounded_deterministic_transform_plus_runtime_checks')
        measured.append({'accepted':record.get('accepted',False),'verified_ir':ir,
            'strategy_id':f"resource.seed{record['seed']}.{record['variant']}",
            'candidate_code_dir':str(folder),'source_phase':'deterministic_resource_unlock',
            'path':ir.get('strategy',{}).get('applied_strategy_ids',[]),
            'path_code':[int(n) for n in ir.get('strategy',{}).get('chain_code','').split('-') if n.isdigit()]})
    top=build_top_terminal_results(measured,3)
    report={'selection_mode':'resumed_bounded_resource_unlock','results':top,'report':str(output/'report.json'),
        'scope':'repaired Phase 1 seeds; bounded deterministic unlock, not a new LLM search',
        'includes_remeasured_controls':True}
    save_json(root/'results/check/bounded_unlock_top3.json',report)
    save_json(root/'results/check/top_3_terminal_results.json',report)
    return top


def main():
    root=Path(__file__).resolve().parents[1]
    report=load_json(root/'results/check/targeted_repair_top3.json')
    report_dir=Path(report['source_report']).parent
    candidates=[]
    for record in report['results']:
        folder=Path(record['code_dir'])
        ir=load_json(report_dir/folder.name/'verified_ir.json')
        candidates.append({'accepted':True,'strategy_id':'repaired.'+folder.name,
            'verified_ir':ir,'candidate_code_dir':str(folder),
            'path':ir.get('strategy',{}).get('applied_strategy_ids',[]),
            'source_snapshot':{name:(folder/name).read_text(encoding='utf-8') for name in ('main.cpp','kernel.h','cuda_kernel.cuh')}})
    output=root/'results/code'/('bounded_unlock_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run_bounded_unlock(candidates,output)
    top=publish_report(root,output)
    print(top,flush=True)


if __name__=='__main__':
    main()
