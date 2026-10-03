"""Compile/run frozen failures in new directories; never overwrite input chains."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from SCOPE.app import repair_chain_locally, verify_terminal_chain_once, terminal_chain_is_accepted
from SCOPE.utils.common_utils import load_json, save_json
from SCOPE.verification.locked_repair import locked_tile_parameters


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/code' / ('fixed_repair_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    library = load_json(root / 'data/lib/strategy_library.json')
    records = []
    for case in sorted((root / 'tests/fixtures/repair_20260916').glob('case*')):
        target = output / case.name
        shutil.copytree(case, target)
        ir = load_json(case / 'failed_ir.json')
        repair = repair_chain_locally(target, target, ir=ir, client=None, strategy_library=library)
        record = {'case': case.name, 'code_dir': str(target), 'repair': repair}
        if repair['status'] == 'repair_generated':
            ir['locked_repair_contract'] = {
                'strategy_ids': ir['strategy']['applied_strategy_ids'],
                'tile_parameters': locked_tile_parameters(ir, (case/'cuda_kernel.cuh').read_text(encoding='utf-8'))}
            verified = verify_terminal_chain_once(ir, target, 'regression.'+case.name, library, 1,
                                                  benchmark_runs=5, benchmark_warmup_runs=2)
            save_json(target/'verified_ir.json', verified)
            record['accepted'] = terminal_chain_is_accepted(verified)
            record['verification'] = verified.get('verification', {}).get('summary', {})
            repair['status'] = 'repair_verified' if record['accepted'] else 'repair_validation_failed'
        records.append(record)
        save_json(output/'report.json', {'records': records, 'inputs_unchanged': True})
        print(json.dumps(record.get('verification', repair), ensure_ascii=False), flush=True)
    print('REPORT: '+str(output/'report.json'), flush=True)
    if len(records) != 3 or not all(r.get('accepted') for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
