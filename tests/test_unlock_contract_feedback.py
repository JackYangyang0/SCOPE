import copy
import unittest
from unittest.mock import Mock

from SCOPE.app import record_unlock_feedback
from SCOPE.generate_ir.unlock_feedback import choose_unlock_steps, infrastructure_failure
from SCOPE.llm.patch_contract import normalize_patch_contract, PatchContractError
from SCOPE.llm.patch_generator import validate_patch_scope, generate_patch_with_llm
from SCOPE.verification.memory_access_plan import vector_mapping_is_proven


class UnlockContractFeedbackTests(unittest.TestCase):
    def strategy(self):
        return {
            'strategy_id': 'Vectorization.GlobalLoadAB.float4',
            'postconditions': ['vectorization.A.vector_width != null',
                               'vectorization.B.vector_width != null'],
            'patch_contract': {'allowed_regions': ['GLOBAL_TO_SHARED_LOAD']},
        }

    def patch(self):
        return {'ir_updates': {'vectorization.A.vector_width': 4},
                'modified_code_regions': [{'file': 'cuda_kernel.cuh',
                                          'region': 'GLOBAL_TO_SHARED_LOAD'}]}

    def test_partial_contract_infers_only_declared_outputs(self):
        strategy = self.strategy()
        original = copy.deepcopy(strategy)
        normalized = normalize_patch_contract(strategy)
        self.assertEqual(normalized['patch_contract']['allowed_ir_fields'],
                         ['vectorization.A.vector_width', 'vectorization.B.vector_width'])
        validate_patch_scope(self.patch(), strategy)
        self.assertEqual(strategy, original)

    def test_explicit_empty_scope_remains_authoritative(self):
        strategy = self.strategy()
        strategy['patch_contract']['allowed_ir_fields'] = []
        with self.assertRaisesRegex(ValueError, 'protected IR'):
            validate_patch_scope(self.patch(), strategy)

    def test_protected_input_and_code_region_rejected(self):
        patch = self.patch()
        patch['ir_updates'] = {'hardware.sm_count': 1}
        with self.assertRaisesRegex(ValueError, 'protected IR'):
            validate_patch_scope(patch, self.strategy())
        patch = self.patch()
        patch['modified_code_regions'][0]['region'] = 'STORE'
        with self.assertRaisesRegex(ValueError, 'protected code'):
            validate_patch_scope(patch, self.strategy())

    def test_structured_post_does_not_authorize_rhs_inputs(self):
        strategy = self.strategy()
        strategy['postconditions'] = {'patch_ir_verification': {'predicates': [
            {'lhs': {'field': 'vectorization.A.vector_width'},
             'rhs': {'field': 'hardware.warp_size'}},
            {'lhs': {'field': 'hardware.sm_count'}}]}}
        self.assertEqual(normalize_patch_contract(strategy)['patch_contract']['allowed_ir_fields'],
                         ['vectorization.A.vector_width'])

    def test_malformed_contract_fails_before_api_call(self):
        client = Mock()
        strategy = self.strategy()
        strategy['patch_contract']['allowed_regions'] = 'STORE'
        with self.assertRaises(PatchContractError):
            generate_patch_with_llm(client, {}, strategy, {}, {})
        client.complete_json.assert_not_called()

    def test_pending_steps_execute_before_new_proposals(self):
        steps = [{'batch_id': str(i), 'strategy_ids': [str(i)]} for i in range(5)]
        first, pending, _ = choose_unlock_steps([], steps, set('01234'), {}, 3)
        second, pending, _ = choose_unlock_steps(pending, steps, set('01234'),
                                                 {'applied_strategy_ids': list('012')}, 3)
        self.assertEqual([x['batch_id'] for x in first], list('012'))
        self.assertEqual([x['batch_id'] for x in second], list('34'))
        self.assertEqual(pending, [])

    def test_pending_eligibility_is_rechecked(self):
        steps = [{'strategy_ids': [s]} for s in 'abc']
        selected, pending, rejected = choose_unlock_steps(steps, [], {'a', 'b'},
                                                           {'failed_unlock_bundles': ['a']}, 1)
        self.assertEqual(selected, [{'strategy_ids': ['b']}])
        self.assertEqual(pending, [])
        self.assertEqual(len(rejected), 2)

    def test_infrastructure_failure_does_not_poison_search(self):
        state = {'history': {}}
        result = {'summary': {'last_verification': {'error_type': 'KeyError'}}}
        self.assertTrue(infrastructure_failure(result['summary']))
        record_unlock_feedback(state, {'strategy_ids': ['float4']}, result, 1)
        self.assertNotIn('failed_strategy_counts', state['history'])
        self.assertNotIn('failed_unlock_bundles', state['history'])
        self.assertEqual(len(state['history']['infrastructure_failures']), 1)

    def test_real_compile_failure_is_still_recorded(self):
        state = {'history': {}}
        record_unlock_feedback(state, {'strategy_ids': ['float4']},
                               {'summary': {'status': 'failed', 'compile_status': 'fail'}}, 1)
        self.assertEqual(state['history']['failed_strategy_counts'], {'float4': 1})

    def test_const_tile_alias_preserves_vector_mapping_proof(self):
        source = '''/* NEXT_TILE_LOAD_BEGIN */
        const int k_base = bkIdx * BK;
        for (int v = tid; v < BM * (BK / 4); v += thread_num) {
            const int ak = (v % (BK / 4)) * 4;
            const int gm = tile_m0 + am;
            const int gk = k_base + ak;
            if (gm < M && gk + 3 < K) { float4 av = FLOAT4(A[OFFSET(gm,gk,K)]); }
        }
        for (int v = tid; v < BK * (BN / 4); v += thread_num) {
            const int bn = (v % (BN / 4)) * 4;
            const int gk = k_base + bk;
            const int gn = tile_n0 + bn;
            if (gk < K && gn + 3 < N) { float4 bv = FLOAT4(B[OFFSET(gk,gn,N)]); }
        }
        /* NEXT_TILE_LOAD_END */'''
        self.assertTrue(vector_mapping_is_proven(source))
        self.assertFalse(vector_mapping_is_proven(source.replace('k_base + ak', 'k_base + ak + 1')))
        self.assertFalse(vector_mapping_is_proven(source.replace('bkIdx * BK;', 'bkIdx * BK + 1;')))
        self.assertFalse(vector_mapping_is_proven(source.replace('const int k_base', 'int k_base')))


if __name__ == '__main__':
    unittest.main()
