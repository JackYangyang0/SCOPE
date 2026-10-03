import unittest

from SCOPE.app import select_stage_survivors, tuning_observations
from SCOPE.verification.strategy_application import (
    strategy_application, update_strategy_contract_state, unproven_dependencies,
)
from SCOPE.verification.strategy_realization_oracle import _check_strategy


class ContractGuidedSearchTests(unittest.TestCase):
    def test_deterministic_tile_has_parameter_evidence(self):
        source = 'void cuda_gemm() { static const int BM = 128; static const int BN = 64; static const int BK = 32; }'
        self.assertEqual(_check_strategy('Tiling.BlockTile.128x64x32', source, '')['status'], 'realized')
        self.assertEqual(_check_strategy('Tiling.BlockTile.64x64x32', source, '')['status'], 'not_realized')

    def test_store_downgrade_requires_repair_without_rejecting_runtime(self):
        sid = 'Vectorization.StoreC.float4'
        report = _check_strategy(sid, '*reinterpret_cast<float2*>(&C[idx]) = out;', '')
        self.assertEqual(report['status'], 'degraded')
        ir = {'verification': {'accepted': True}, 'strategy': {'applied_strategy_ids': [sid]},
              'strategy_realization': {'strategy_reports': [report]}}
        self.assertTrue(strategy_application(ir, [sid])['repair_required'])
        self.assertTrue(ir['verification']['accepted'])
        state = update_strategy_contract_state(ir)
        self.assertEqual(state['realized_capabilities'], [])
        self.assertEqual(state['pending_obligations'][0]['status'], 'degraded')

    def test_obligations_survive_missing_history_entry(self):
        ir = {'strategy': {'applied_strategy_ids': ['old']}}
        update_strategy_contract_state(ir)
        ir['strategy']['applied_strategy_ids'] = ['new']
        state = update_strategy_contract_state(ir)
        self.assertEqual(state['selected_strategies'], ['old', 'new'])
        self.assertEqual(len(state['pending_obligations']), 2)

    def test_unknown_does_not_satisfy_dependency(self):
        ir = {'strategy': {'applied_strategy_ids': ['Vectorization.StoreC.float4']}}
        update_strategy_contract_state(ir)
        self.assertEqual(unproven_dependencies(ir, ['Vectorization.StoreC.*']), ['Vectorization.StoreC.*'])

    def test_any_dependency_can_use_realized_alternative(self):
        ir = {'strategy_contract_state': {'selected_strategies': ['a', 'b'], 'realized_capabilities': ['b']}}
        self.assertEqual(unproven_dependencies(ir, [{'mode': 'any', 'strategies': ['a', 'b']}]), [])

    def test_known_runtime_candidates_preserve_different_blocks(self):
        states = []
        for index, (bm, bn, score) in enumerate([(128, 64, 100), (128, 64, 99), (64, 64, 20), (64, 128, 30)]):
            states.append({'path_code': [index+1], 'current_ir': {
                'tiling': {'block_m': bm, 'block_n': bn, 'block_k': 16},
                'resource': {'tiling_candidate': {'architecture_score': score}}}})
        kept, removed = select_stage_survivors(states, 3)
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0], states[0])
        self.assertIn(states[2], kept)
        self.assertIn(states[3], kept)
        self.assertEqual(removed, [states[1]])

    def test_low_cta_is_feedback_not_a_rejection(self):
        ir = {'problem': {'M': 512, 'N': 512, 'K': 512},
              'tiling': {'block_m': 128, 'block_n': 64, 'block_k': 32},
              'hardware': {'sm_count': 34}}
        report = tuning_observations(ir)
        self.assertTrue(any('Fewer CTAs' in s for s in report['suggested_controlled_comparisons']))


if __name__ == '__main__':
    unittest.main()
