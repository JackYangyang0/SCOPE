import copy
import json
import unittest
from pathlib import Path

from SCOPE.llm.patch_generator import load_strategy, validate_patch_scope
from SCOPE.llm.unlock_category_selector import category_of
from SCOPE.verification.load_layout_trials import transform, update_trial_ir
from SCOPE.verification.unlock_evidence import classify_diagnosis


class LoadLayoutTrialTests(unittest.TestCase):
    def test_prefetch_declares_narrow_resource_permission(self):
        library = json.loads((Path(__file__).parents[1] / 'data/lib/strategy_library.json').read_text(encoding='utf-8'))
        strategy = load_strategy(library, 'Pipeline.WarpRegisterPrefetchAB')
        strategy['patch_contract'] = {'allowed_regions': ['REGISTER_DECL']}
        patch = {'ir_updates': {'resource.register.prefetch_slots': 2},
                 'modified_code_regions': [{'file': 'cuda_kernel.cuh', 'region': 'REGISTER_DECL'}]}
        validate_patch_scope(patch, strategy)
        patch['ir_updates'] = {'hardware.sm_count': 1}
        with self.assertRaises(ValueError):
            validate_patch_scope(patch, strategy)

    def test_store_aliases_share_category(self):
        for sid in ('Vectorization.StoreC.float4', 'Vectorization.StoreC.GuardedVectorStore', 'Epilogue.StoreC.Vectorized.float4'):
            self.assertEqual(category_of({'strategy_id': sid}), 'epilogue')

    def test_layout_override_records_superseded_strategy(self):
        ir = {'strategy': {'applied_strategy_ids': ['Layout.SharedMemory.PaddingAB.Plus1', 'Tiling.BlockTile.64x64x16']}}
        layout = {'shared_A': {'padding': 0, 'shape': [2, 16, 64]},
                  'shared_B': {'padding': 0, 'shape': [2, 16, 64]}, 'shared_memory_bytes': 16384}
        removed = update_trial_ir(ir, 'none', layout)
        self.assertEqual(removed, ['Layout.SharedMemory.PaddingAB.Plus1'])
        self.assertEqual(ir['strategy']['applied_strategy_ids'], ['Tiling.BlockTile.64x64x16'])
        self.assertEqual(ir['resource']['shared_memory']['total_bytes'], 16384)

    def test_proof_gap_retains_evidence_without_claiming_failure(self):
        diagnosis = {'defects': [{'evidence': {'message': 'Vectorized access lacks alignment guard/proof: vectorization.C.'}}]}
        original = copy.deepcopy(diagnosis)
        result = classify_diagnosis(diagnosis, {k: 'pass' for k in ('compile_status', 'correctness_status', 'runtime_safety_status')})
        self.assertEqual(result['defects'][0]['selection_evidence_kind'], 'static_proof_gap')
        self.assertEqual(diagnosis, original)
        self.assertTrue(result['defects'][0]['tested_execution_passed'])

    def test_unsupported_source_is_not_replaced(self):
        with self.assertRaises((ValueError, KeyError)):
            transform('void cuda_gemm() {}', 'none_vector_b_fixed')


if __name__ == '__main__':
    unittest.main()
