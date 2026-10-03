import unittest
from SCOPE.verification.strategy_application import strategy_application
from SCOPE.app import terminal_chain_is_accepted


class StrategyApplicationTests(unittest.TestCase):
    def ir(self, status):
        return {'verification': {'accepted': True}, 'strategy_realization': {
            'strategy_reports': [{'strategy_id': 'new', 'status': status}]}}

    def test_missing_is_repairable_but_runtime_candidate_survives(self):
        ir = self.ir('not_realized')
        report = strategy_application(ir, ['new'])
        self.assertTrue(terminal_chain_is_accepted(ir))
        self.assertTrue(report['repair_required'])
        self.assertFalse(report['dependency_update_allowed'])

    def test_unknown_is_not_implementation_failure(self):
        for status in ('unknown', 'partially_realized'):
            report = strategy_application(self.ir(status), ['new'])
            self.assertEqual(report['status'], 'unproven')
            self.assertFalse(report['repair_required'])
            self.assertFalse(report['dependency_update_allowed'])
            self.assertTrue(report['continuation_allowed'])
            self.assertEqual(report['realized_strategy_ids'], [])

    def test_realized_can_advance(self):
        self.assertTrue(strategy_application(self.ir('realized'), ['new'])['dependency_update_allowed'])

    def test_lost_previous_optimization_requires_repair(self):
        ir = self.ir('realized')
        ir['strategy_realization']['strategy_reports'].append({'strategy_id': 'old', 'status': 'not_realized'})
        report = strategy_application(ir, ['new'])
        self.assertTrue(report['repair_required'])
        self.assertEqual(report['missing_preserved_strategy_ids'], ['old'])

    def test_absent_report_is_unknown(self):
        self.assertEqual(strategy_application({}, ['new'])['status'], 'unproven')

    def test_runtime_failure_cannot_continue(self):
        ir = self.ir('unknown')
        ir['verification']['accepted'] = False
        self.assertFalse(strategy_application(ir, ['new'])['continuation_allowed'])

    def test_known_missing_cannot_continue(self):
        self.assertFalse(strategy_application(self.ir('not_realized'), ['new'])['continuation_allowed'])
