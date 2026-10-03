import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.app import repair_chain_locally
from SCOPE.verification.locked_repair import locked_tile_parameters, tile_lock_defects


SOURCE = '''// outside scoped regions: full context must contain this
static const int BM = 64;
static const int BN = 64;
static const int BK = 16;
/* STORE_BEGIN */
old_store();
/* STORE_END */
'''


class LockedRepairTests(unittest.TestCase):
    def test_strategy_parameters_override_wrong_generated_values(self):
        ir = {'strategy': {'applied_strategy_ids': ['Tiling.BlockTile.128x64x16']}}
        locked = locked_tile_parameters(ir, SOURCE)
        self.assertEqual(locked['BM'], 128)
        self.assertTrue(tile_lock_defects(SOURCE, locked))

    def run_repair(self, *, mutate_tile=False, reselect=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cuda_kernel.cuh').write_text(SOURCE)
            (root / 'kernel.h').write_text('void cuda_gemm();')
            (root / 'main.cpp').write_text('BENCHMARK_MUST_NOT_BE_SENT')
            ir = {'problem': {'target': 'GPU'}, 'strategy': {
                'applied_strategy_ids': ['Tiling.BlockTile.64x64x16'],
                'history': [{'stage': 'Tiling'}]}}
            original_ir = copy.deepcopy(ir)
            response = {'edits': [{'path': 'cuda_kernel.cuh', 'region': 'STORE', 'replacement': 'new_store();'}]}
            if reselect:
                response['selected_strategy_ids'] = []
            def apply(*args):
                text = SOURCE.replace('old_store', 'new_store')
                if mutate_tile:
                    text = text.replace('BM = 64', 'BM = 32')
                (root / 'cuda_kernel.cuh').write_text(text)
                return {'status': 'pass'}
            with patch('SCOPE.app.apply_deterministic_compile_repairs', side_effect=lambda s,d:s), \
                 patch('SCOPE.app.generate_code_files_from_patch_with_llm', return_value=response) as generate, \
                 patch('SCOPE.app.apply_generated_code_files', side_effect=apply), \
                 patch('SCOPE.app.check_generated_source_syntax', return_value={'accepted': True}):
                result = repair_chain_locally(root, root, {'stage': 'Chain'}, ir,
                    client=object(), baseline_snapshot={'cuda_kernel.cuh': 'older baseline'})
            context = generate.call_args.kwargs
            self.assertEqual(context['code_context']['files']['cuda_kernel.cuh'], SOURCE)
            self.assertNotIn('main.cpp', context['code_context']['files'])
            self.assertEqual(context['prompt_path'].name, 'repair_locked_cuda_prompt.txt')
            self.assertIn('changes_since_repair_baseline', context['repair_context'])
            self.assertEqual(ir, original_ir)
            if mutate_tile or reselect:
                self.assertEqual((root / 'cuda_kernel.cuh').read_text(), SOURCE)
            return result

    def test_full_context_without_restoring_stale_baseline(self):
        self.assertEqual(self.run_repair()['status'], 'repair_generated')

    def test_tile_change_rolls_back(self):
        self.assertEqual(self.run_repair(mutate_tile=True)['status'], 'repair_failed')

    def test_strategy_reselection_is_rejected(self):
        self.assertEqual(self.run_repair(reselect=True)['failure_class'],
                         'implementation_repair_failed_not_strategy_conflict')


if __name__ == '__main__':
    unittest.main()
