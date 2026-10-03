import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock
from SCOPE.specialization.tile_range_planner import expand_plan, plan_tile_ranges


class RangePlannerTests(unittest.TestCase):
    def test_raw_parse_failure_is_saved_and_corrected_once(self):
        class Client:
            def __init__(self):
                self.calls = []
            def complete_tile_plan_text(self, messages):
                self.calls.append(list(messages))
                return '{"groups":[' if len(self.calls) == 1 else '{"groups":[]}'
        client = Client()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(plan_tile_ranges(client, {}, 'kernel', audit_dir=tmp), {'groups': []})
            self.assertEqual(len(client.calls), 2)
            self.assertIn('Validation failed', client.calls[1][-1]['content'])
            raw = json.loads((Path(tmp) / 'attempt_1.response.json').read_text())
            self.assertEqual(raw['raw_content'], '{"groups":[')
            status = json.loads((Path(tmp) / 'planning_status.json').read_text())
            self.assertEqual(status['status'], 'planned')

    def test_excess_groups_retried_without_library_fallback(self):
        client = Mock()
        client.complete_json.side_effect = [{'groups': [self.group()] * 84}, {'groups': [self.group()]}]
        result = plan_tile_ranges(client, {}, 'kernel')
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual(client.complete_json.call_count, 2)

    def test_persistent_error_stops_after_two_calls(self):
        client = Mock()
        client.complete_json.return_value = {'groups': [self.group()] * 84}
        with self.assertRaisesRegex(ValueError, 'Too many'):
            plan_tile_ranges(client, {}, 'kernel')
        self.assertEqual(client.complete_json.call_count, 2)

    def group(self, block=64, warp=32):
        return {'block_tiles': [[block, block, 16]], 'warp_tiles': [[warp, warp]], 'thread_tiles': [[2, 2]]}

    def test_groups_do_not_cross_and_duplicates_are_removed(self):
        a, b = self.group(), self.group(128, 64)
        merged = expand_plan({'groups': [a, b, a]}, {})
        self.assertTrue(merged)
        self.assertEqual(len(merged), len(expand_plan({'groups': [a]}, {})) + len(expand_plan({'groups': [b]}, {})))
        self.assertTrue(all((t['BM'], t['WM']) in {(64, 32), (128, 64)} for t in merged))

    def test_empty_plan_never_falls_back(self):
        self.assertEqual(expand_plan({'groups': []}, {}), [])

    def test_illegal_mapping_filtered(self):
        self.assertEqual(expand_plan({'groups': [self.group(32, 64)]}, {}), [])

    def test_invalid_and_oversized_plans_fail(self):
        with self.assertRaises(ValueError):
            expand_plan({'groups': [self.group()]}, {}, 1)
        group = self.group()
        group['thread_tiles'] = [[True, 2]]
        with self.assertRaises(ValueError):
            expand_plan({'groups': [group]}, {})

    def test_prompt_uses_current_kernel_and_ir(self):
        client = Mock()
        client.complete_json.return_value = {'groups': []}
        plan_tile_ranges(client, {'problem': {'M': 1024}}, 'current kernel')
        messages = client.complete_json.call_args.args[0]
        self.assertIn('current kernel', messages[1]['content'])
        self.assertIn('1024', messages[1]['content'])


if __name__ == '__main__':
    unittest.main()
