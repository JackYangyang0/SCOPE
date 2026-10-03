import unittest
from SCOPE.app import build_top_terminal_results, merge_terminal_verifications


class TerminalRankingAcceptanceTests(unittest.TestCase):
    def test_rejected_fast_result_not_published_as_winner(self):
        rejected = {'strategy_id': 'rejected', 'accepted': False, 'verified_ir': {
            'verification': {'summary': {'compile_status': 'pass', 'correctness_status': 'pass'}},
            'performance': {'gflops': 10000}}}
        self.assertEqual(build_top_terminal_results([rejected], 3), [])
        merged = merge_terminal_verifications({'verified_candidates': [rejected]})
        self.assertIsNone(merged['best_candidate'])
        self.assertEqual(len(merged['verified_candidates']), 1)


if __name__ == '__main__':
    unittest.main()
