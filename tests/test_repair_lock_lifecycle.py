import copy
import unittest
from SCOPE.verification.locked_repair import begin_optimization_transaction


class RepairLockLifecycleTests(unittest.TestCase):
    def test_new_optimization_retires_lock_without_mutating_parent(self):
        old = {'strategy': {'applied_strategy_ids': ['A']},
               'locked_repair_contract': {'strategy_ids': ['A'], 'tile_parameters': {'BM': 64}},
               'locked_repair_verification': {'status': 'pass'}}
        original = copy.deepcopy(old)
        new = begin_optimization_transaction(old)
        new['strategy']['applied_strategy_ids'].append('B')
        self.assertNotIn('locked_repair_contract', new)
        self.assertNotIn('locked_repair_verification', new)
        self.assertEqual(new['repair_lock_history'][0]['contract']['strategy_ids'], ['A'])
        self.assertEqual(old, original)

    def test_no_lock_keeps_other_constraints(self):
        ir = {'strategy': {'applied_strategy_ids': ['A']}, 'hard_constraints': ['must_preserve_K']}
        self.assertEqual(begin_optimization_transaction(ir), ir)
