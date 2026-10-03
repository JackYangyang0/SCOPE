import tempfile
import unittest
from pathlib import Path
from SCOPE.llm.concrete_code_generator import compose_nested_region_edits
from SCOPE.verification.locked_repair import cooperative_load_reference, vector_coverage_hints
from SCOPE.verification.optimization_preservation import staged_buffers


class LiveRepairFeedbackTests(unittest.TestCase):
    def test_partial_vector_coverage_is_actionable_not_name_based(self):
        source = 'static const int BM=64;static const int BN=128;static const int BK=16;static const int WM=16;static const int WN=64;'
        source += '/* GLOBAL_TO_SHARED_LOAD_BEGIN */float4 t=FLOAT4(B[addr]);/* GLOBAL_TO_SHARED_LOAD_END */'
        hints = vector_coverage_hints(source, {})
        self.assertEqual(hints[0]['required_tile_elements'], 2048)
        self.assertEqual(hints[0]['vector_fast_path_upper_bound'], 1024)
        self.assertEqual(vector_coverage_hints(source.replace('float4 t=FLOAT4(B[addr]);', 'for(int v=tid;v<512;v+=256){float4 t=FLOAT4(B[addr]);}'), {}), [])
    def test_literal_stage_constant_does_not_drop_buffer(self):
        source = 'constexpr int STAGES=2; __shared__ float As[STAGES][BK][BM];'
        self.assertEqual(staged_buffers(source), {'As':2})
        self.assertEqual(staged_buffers('__shared__ float As[UNKNOWN][BK][BM];'), {})
    def test_compose_nested_patch_without_dropping_child(self):
        body = '/* COMPUTE_INNER_BEGIN */old;/* COMPUTE_INNER_END */'
        source = '/* MAIN_LOOP_BEGIN */'+body+'/* MAIN_LOOP_END */'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'cuda_kernel.cuh').write_text(source)
            edits = [{'path':'cuda_kernel.cuh','region':'MAIN_LOOP','replacement':body},
                     {'path':'cuda_kernel.cuh','region':'COMPUTE_INNER','replacement':'fixed;'}]
            result = compose_nested_region_edits(edits, root)
            self.assertEqual(len(result), 1)
            self.assertIn('fixed;', result[0]['replacement'])
            self.assertNotIn('old;', result[0]['replacement'])
            edits[0]['replacement'] = 'no_anchors;'
            with self.assertRaises(ValueError):
                compose_nested_region_edits(edits, root)

    def test_reference_retains_vector_loads_but_scatter_is_scalar(self):
        source = '__shared__ float As[2][BK][BM]; __shared__ float Bs[2][BK][BN];'
        ref = cooperative_load_reference(source, {'problem':{'layout_A':'row_major','trans_A':False}})
        self.assertIn('v += threads', ref['reference'])
        self.assertIn('FLOAT4(B[', ref['reference'])
        self.assertIn('As[stage][ak + 3][am]', ref['reference'])
        self.assertNotIn('FLOAT4(Bs[', ref['reference'])


if __name__ == '__main__':
    unittest.main()
