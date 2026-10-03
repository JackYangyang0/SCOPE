import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from SCOPE.llm.concrete_code_generator import replace_cuda_anchor_region, apply_generated_code_files
from SCOPE.verification.optimization_preservation import preservation_defects


class PatchPreservationTests(unittest.TestCase):
    def test_application_rejects_downgrade_without_writing(self):
        source = '/* GLOBAL_TO_SHARED_LOAD_BEGIN */\nfloat4 v = FLOAT4(A[i]);\n/* GLOBAL_TO_SHARED_LOAD_END */'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'cuda_kernel.cuh'
            path.write_text(source)
            edits = {'strategy_id': 'Reordering.ThreadMapping.CoalescedLoadA', 'edits': [
                {'path': 'cuda_kernel.cuh', 'region': 'GLOBAL_TO_SHARED_LOAD', 'replacement': 'float v=A[i];'}]}
            with patch('SCOPE.llm.concrete_code_generator.ensure_generated_code_has_minimal_kernel'):
                result = apply_generated_code_files(edits, root, {}, {'strategy_id': edits['strategy_id']})
            self.assertEqual(result['status'], 'fail')
            self.assertEqual(path.read_text(), source)

    def test_double_buffer_is_a_core_candidate(self):
        from SCOPE.app import infer_strategy_phase, PHASE1_CORE_COUPLED
        self.assertEqual(infer_strategy_phase('Pipeline.DoubleBuffer.SharedAB'), PHASE1_CORE_COUPLED)

    def test_single_line_anchor_replaces_not_appends(self):
        source = 'prefix\n/* NEXT_TILE_LOAD_BEGIN */\nold_load();\n/* NEXT_TILE_LOAD_END */\nsuffix\n'
        result = replace_cuda_anchor_region(source, 'NEXT_TILE_LOAD', 'vector_load();')
        self.assertNotIn('old_load', result)
        self.assertEqual(result.count('vector_load();'), 1)
        self.assertEqual(replace_cuda_anchor_region(result, 'NEXT_TILE_LOAD', 'vector_load();'), result)
        self.assertTrue(result.endswith('suffix\n'))

    def test_multiline_and_nested_markers(self):
        source = '/*\n * MAIN_LOOP_BEGIN\n */\n/* NEXT_TILE_LOAD_BEGIN */\nx();\n/* NEXT_TILE_LOAD_END */\n/*\n * MAIN_LOOP_END\n */'
        result = replace_cuda_anchor_region(source, 'NEXT_TILE_LOAD', 'y();')
        self.assertNotIn('x();', result)
        self.assertIn('MAIN_LOOP_END', result)

    def test_duplicate_markers_fail(self):
        source = '/* STORE_BEGIN */ x(); /* STORE_END */\n' * 2
        with self.assertRaises(ValueError):
            replace_cuda_anchor_region(source, 'STORE', 'y();')

    def test_repair_cannot_silently_downgrade(self):
        before = '__shared__ float As[2][BK][BM]; float4 v = FLOAT4(A[i]);'
        after = '__shared__ float As[BK][BM]; float v = A[i];'
        self.assertEqual(len(preservation_defects(before, after, {'strategy_id': 'Repair.PreserveAppliedStrategies'})), 2)

    def test_macro_or_comment_does_not_count_as_vector_load(self):
        before = '#define FLOAT4(x) foo(x)\n// FLOAT4(A[i])\nfloat x=A[i];'
        self.assertEqual(preservation_defects(before, 'float x=A[i];', {}), [])

    def test_no_async_is_not_permission_to_remove_double_buffer(self):
        self.assertTrue(preservation_defects('__shared__ float As[2][BK][BM];',
            '__shared__ float As[BK][BM];', {'strategy_id': 'Pipeline.NoAsyncCopy.V1'}))


if __name__ == '__main__':
    unittest.main()
