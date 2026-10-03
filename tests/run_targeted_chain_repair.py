"""Explicit, backed-up repair/reverification of terminal chains (no LLM)."""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from SCOPE.app import repair_chain_locally, verify_terminal_chain_once, terminal_chain_is_accepted
from SCOPE.utils.common_utils import load_json, save_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Modify chain files after making source backups')
    parser.add_argument('--verify-unchanged', action='store_true', help='Also reverify chains without a matching repair')
    args=parser.parse_args()
    output=ROOT/'results'/'code'/('targeted_repair_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    library=load_json(ROOT/'data/lib/strategy_library.json')
    records=[]
    for ir_path in sorted((ROOT/'data/IRs/ir_patch').glob('optir.verified.Chain.*.attempt_1.json')):
        ir=load_json(ir_path)
        code=ir.get('strategy',{}).get('chain_code')
        if not code:
            continue
        source=ROOT/'gemm_code/code'/('chain.'+code)
        if not source.exists():
            continue
        backup=output/source.name/'original'
        backup.mkdir(parents=True)
        for name in ('cuda_kernel.cuh','kernel.h','main.cpp'):
            shutil.copy2(source/name,backup/name)
        target=source if args.apply else output/source.name/'repaired'
        result=repair_chain_locally(source,target,ir.get('defect_diagnosis',{}),ir,
                                   client=None,strategy_library=library)
        record={'source':str(source),'code_dir':str(target),'backup':str(backup),'repair':result}
        if result['status']=='repair_generated' or (args.verify_unchanged and result['status']=='repair_unavailable'):
            from SCOPE.verification.locked_repair import locked_tile_parameters
            if not target.exists():
                shutil.copytree(backup,target)
            ir['locked_repair_contract']={'strategy_ids':ir.get('strategy',{}).get('applied_strategy_ids',[]),
                                         'tile_parameters':locked_tile_parameters(ir,(target/'cuda_kernel.cuh').read_text(encoding='utf-8'))}
            verified=verify_terminal_chain_once(ir,target,'targeted.'+code,library,1,
                                               benchmark_runs=5,benchmark_warmup_runs=2)
            save_json(output/source.name/'verified_ir.json',verified)
            record.update(accepted=terminal_chain_is_accepted(verified),
                          verification=verified.get('verification',{}).get('summary',{}),
                          memory_checks=verified.get('memory_access_plan_verification'),
                          locked_repair=verified.get('locked_repair_verification'))
        records.append(record)
        save_json(output/'report.json',{'records':records,'historical_ranking_overwritten':False})
        print(json.dumps({k:record.get(k) for k in ('source','accepted','verification')},ensure_ascii=False),flush=True)
    top=sorted((r for r in records if r.get('accepted')),key=lambda r:r['verification'].get('gflops') or 0,reverse=True)
    save_json(ROOT/'results/check/targeted_repair_top3.json',{'source_report':str(output/'report.json'),'results':top[:3]})
    print('REPORT: '+str(output/'report.json'),flush=True)


if __name__=='__main__':
    main()
