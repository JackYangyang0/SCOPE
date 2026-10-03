import unittest

from SCOPE.llm.unlock_category_selector import select_category_plan
from SCOPE.llm.implementation_context import kernel_implementation_summary


class Client:
    def __init__(self, selected):
        self.selected = selected
        self.messages = None

    def complete_selection_json(self, messages):
        self.messages = messages
        return {'selected_strategy_id': self.selected, 'reason': 'test'}


class CategorySelectionTests(unittest.TestCase):
    def select(self, client, strategies, visits=None, **kwargs):
        return select_category_plan(client, {'strategies': strategies},
                                    visits if visits is not None else {}, {}, **kwargs)

    def test_only_current_category_is_sent(self):
        client = Client('Layout.A')
        plan = self.select(client, [{'strategy_id': 'Layout.A'}, {'strategy_id': 'Pipeline.B'}])
        self.assertEqual(plan['category'], 'layout')
        self.assertNotIn('Pipeline.B', client.messages[1]['content'])

    def test_skip_does_not_force_fallback(self):
        self.assertEqual(self.select(Client(None), [{'strategy_id': 'Layout.A'}])['batches'], [])

    def test_cross_category_selection_rejected(self):
        with self.assertRaises(ValueError):
            self.select(Client('Pipeline.B'), [{'strategy_id': 'Layout.A'}, {'strategy_id': 'Pipeline.B'}])

    def test_dependencies_preserved_atomically(self):
        plan = self.select(Client('Vectorization.A'), [
            {'strategy_id': 'Vectorization.A', 'requires_bundle_strategy_ids': ['Pipeline.B']},
            {'strategy_id': 'Pipeline.B'}])
        self.assertEqual(plan['batches'][0]['strategy_ids'], ['Pipeline.B', 'Vectorization.A'])

    def test_missing_dependency_rejected(self):
        with self.assertRaises(ValueError):
            self.select(Client('Layout.A'), [{'strategy_id': 'Layout.A', 'requires_bundle_strategy_ids': ['missing']}])

    def test_visits_rotate_and_are_bounded(self):
        items = [{'strategy_id': 'Layout.A'}, {'strategy_id': 'Pipeline.B'}]
        visits = {}
        self.assertEqual(self.select(Client(None), items, visits, max_visits=1)['category'], 'layout')
        self.assertEqual(self.select(Client(None), items, visits, max_visits=1)['category'], 'pipeline')
        self.assertIsNone(self.select(Client(None), items, visits, max_visits=1)['category'])

    def test_conflicting_companions_rejected(self):
        with self.assertRaises(ValueError):
            self.select(Client('Layout.A'), [
                {'strategy_id': 'Layout.A', 'conflict_group': 'layout', 'requires_bundle_strategy_ids': ['Layout.B']},
                {'strategy_id': 'Layout.B', 'conflict_group': 'layout'}])

    def test_existing_source_can_be_recorded_without_generation_or_false_acceptance(self):
        sid = 'Vectorization.GlobalLoadAB.float4'
        source = 'float4 a = FLOAT4(A[k]); float4 b = FLOAT4(B[k]);'
        summary = {'files': [{'path': 'cuda_kernel.cuh',
                             'implementation_evidence': kernel_implementation_summary(source)}]}
        class RecordClient:
            def complete_selection_json(self, messages):
                return {'selected_strategy_id': sid, 'action': 'record_existing'}
        history = {'applied_strategy_ids': []}
        plan = select_category_plan(RecordClient(), {'strategies': [{'strategy_id': sid}]}, {},
                                    history, code_summary=summary)
        self.assertEqual(plan['batches'], [])
        self.assertFalse(plan['code_generation_required'])
        self.assertEqual(history['applied_strategy_ids'], [])
        self.assertEqual(plan['implementation_observations'][sid][0]['status'], 'observed_not_proven')
        with self.assertRaisesRegex(ValueError, 'without current source evidence'):
            self.select(RecordClient(), [{'strategy_id': sid}], code_summary={})
        declaration_only = {'files': [{'implementation_evidence': kernel_implementation_summary('float4 A, B;')}]}
        with self.assertRaises(ValueError):
            self.select(RecordClient(), [{'strategy_id': sid}], code_summary=declaration_only)

    def test_observation_does_not_filter_actual_optimization(self):
        sid = 'Vectorization.GlobalLoadAB.float4'
        summary = {'files': [{'implementation_evidence': kernel_implementation_summary(
            'float4 a = FLOAT4(A[k]); float4 b = FLOAT4(B[k]);')} ]}
        plan = self.select(Client(sid), [{'strategy_id': sid}], code_summary=summary)
        self.assertEqual(plan['batches'][0]['strategy_ids'], [sid])


if __name__ == '__main__':
    unittest.main()
