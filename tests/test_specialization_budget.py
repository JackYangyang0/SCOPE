import unittest
from unittest.mock import patch

from SCOPE.specialization.tile_range_planner import expand_plan
from SCOPE.specialization.top1_shape_transfer import diverse_tiles
from SCOPE.specialization.shape_specialization import current_shape_candidates
from SCOPE.specialization.shape_search import TILE_KEYS


class SpecializationBudgetTests(unittest.TestCase):
    def test_overflow_keeps_legal_candidates_in_bounded_mode(self):
        plan = {'groups': [{'block_tiles': [[64, 64, 16]], 'warp_tiles': [[32, 32]], 'thread_tiles': [[4, 4]]}]}
        tiles = [{k: i + 1 for k in TILE_KEYS} for i in range(4)]
        with patch('SCOPE.specialization.tile_range_planner.enumerate_tiles', return_value=tiles):
            self.assertEqual(expand_plan(plan, {}, 2, allow_bounded=True), tiles[:2])
            with self.assertRaises(ValueError):
                expand_plan(plan, {}, 2)

    def test_block_diversity_before_second_same_block(self):
        a = {'BM': 64, 'BN': 64, 'BK': 16, 'TM': 2}
        b = dict(a, TM=4)
        c = dict(a, BM=128)
        self.assertEqual(diverse_tiles([a, b, c], 2), [a, c])

    def test_other_shapes_never_imported(self):
        report = {'shapes': [{'shape': {'M': 512, 'N': 512, 'K': 512},
                              'top_results': [{'accepted': True, 'code_dir': 'must-not-read'}]}]}
        self.assertEqual(current_shape_candidates(report, {'accepted': True},
                         {'problem': {'M': 1024, 'N': 1024, 'K': 1024}}), [])
