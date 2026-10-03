import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from SCOPE.specialization.terminal_tuning import tune_terminal_candidates
from SCOPE.specialization.shape_search import write_json
from SCOPE.tests.test_shape_search import state, verified


class TerminalTuningTests(unittest.TestCase):
    @patch('SCOPE.specialization.terminal_tuning.plan_tile_ranges', return_value={'groups': []})
    def test_only_verified_seeds_and_failure_isolation(self, planner):
        seeds = [{'accepted': True, 'strategy_id': str(i),
                  'candidate_code_dir': str(i), 'verified_ir': verified(state(), 10-i)} for i in range(2)]
        seeds.append({'accepted': False})
        def tune(sources, kernel, ir, library, output, **kwargs):
            self.assertEqual(kwargs['max_candidates'], 0)
            self.assertEqual(kwargs['candidate_tiles'], [])
            if output.name == 'seed_1':
                raise RuntimeError('test failure')
            output.mkdir()
            (output / kernel).write_text('kernel')
            write_json(output / 'verified_ir.json', verified(ir, 20))
            return {'top_results': [{'code_dir': str(output), 'accepted': True, 'tile': None}],
                    'stop_reason': 'candidate_pool_exhausted'}
        with tempfile.TemporaryDirectory() as tmp, patch('SCOPE.specialization.terminal_tuning.read_sources',
                return_value=(None, 'cuda_kernel.cuh', {'cuda_kernel.cuh': 'kernel'})), patch(
                'SCOPE.specialization.terminal_tuning.tune_sources', side_effect=tune) as runner:
            result = tune_terminal_candidates(seeds, {}, Path(tmp), {}, 'windows', 5, 2)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(len(result['verified_candidates']), 1)
            self.assertEqual(result['summary']['seed_reports'][0]['status'], 'failed')
            self.assertEqual(result['verified_candidates'][0]['source_phase'], 'terminal_tile_tuning')
            self.assertEqual(seeds[0]['verified_ir']['performance']['gflops'], 10)


if __name__ == '__main__':
    unittest.main()
