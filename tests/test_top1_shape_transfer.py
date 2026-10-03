import copy
import tempfile
import unittest
from unittest.mock import patch

from SCOPE.specialization.top1_shape_transfer import transfer_top1


class Top1ShapeTransferTests(unittest.TestCase):
    def test_independent_shapes_and_frozen_source(self):
        seed = {'accepted': True, 'candidate_code_dir': 'seed',
                'verified_ir': {'problem': {'M': 1024, 'N': 1024, 'K': 1024},
                                'performance': {'gflops': 123}}}
        original = copy.deepcopy(seed)
        calls = []

        def tune(sources, kernel, ir, *args, **kwargs):
            calls.append((sources[kernel], dict(ir['problem'])))
            self.assertNotIn('performance', ir)
            sources[kernel] = 'mutated candidate'
            return {'top_results': [{'accepted': True, 'code_dir': 'verified'}]}

        with tempfile.TemporaryDirectory() as folder, \
             patch('SCOPE.specialization.top1_shape_transfer.passed', return_value=True), \
             patch('SCOPE.specialization.top1_shape_transfer.read_sources', return_value=(None, 'cuda_kernel.cuh', {'cuda_kernel.cuh': 'original'})), \
             patch('SCOPE.specialization.top1_shape_transfer.plan_tile_ranges', return_value={}), \
             patch('SCOPE.specialization.top1_shape_transfer.expand_plan', return_value=[]), \
             patch('SCOPE.specialization.top1_shape_transfer.tune_sources', side_effect=tune):
            report = transfer_top1(seed, {}, folder, {'shapes': [512, 2048]}, 'windows', 5, 2)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual([c[0] for c in calls], ['original', 'original'])
        self.assertEqual([c[1]['M'] for c in calls], [512, 2048])
        self.assertEqual(seed, original)

    def test_planning_failure_still_tests_baseline(self):
        seed = {'accepted': True, 'candidate_code_dir': 'seed', 'verified_ir': {'problem': {}}}
        with tempfile.TemporaryDirectory() as folder, \
             patch('SCOPE.specialization.top1_shape_transfer.passed', return_value=True), \
             patch('SCOPE.specialization.top1_shape_transfer.read_sources', return_value=(None, 'cuda_kernel.cuh', {'cuda_kernel.cuh': 'original'})), \
             patch('SCOPE.specialization.top1_shape_transfer.plan_tile_ranges', side_effect=RuntimeError('timeout')), \
             patch('SCOPE.specialization.top1_shape_transfer.tune_sources', return_value={'top_results': []}) as tune:
            report = transfer_top1(seed, {}, folder, {'shapes': [512]}, 'linux', 5, 2)
        self.assertEqual(tune.call_args.kwargs['candidate_tiles'], [])
        self.assertEqual(report['shapes'][0]['planning_error'], 'timeout')
        self.assertEqual(report['status'], 'partial_failure')

    def test_missing_winner_skips(self):
        with tempfile.TemporaryDirectory() as folder:
            report = transfer_top1(None, {}, folder, {}, 'linux', 5, 2)
        self.assertEqual(report['status'], 'skipped_no_verified_top1')
