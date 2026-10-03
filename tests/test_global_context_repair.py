import tempfile
import unittest
from pathlib import Path
from SCOPE.app import load_strategy_scoped_code_context
from SCOPE.utils.common_utils import load_json
from SCOPE.verification.cooperative_load_repair import repair_cooperative_loads
from SCOPE.verification.optimization_preservation import preservation_defects
from SCOPE.verification.memory_access_plan import anchor_body, memory_access_consistency_defects
from SCOPE.verification.gemm_semantic_checker import code_segment_has_k_tile_dependency


class GlobalContextTests(unittest.TestCase):
    def test_nested_anchor_does_not_truncate_parent_loop(self):
        source = '/* MAIN_LOOP_BEGIN */before;/* COMPUTE_INNER_BEGIN */inside;/* COMPUTE_INNER_END */after;/* MAIN_LOOP_END */'
        self.assertIn('inside;', anchor_body(source, 'MAIN_LOOP'))
        self.assertIn('after;', anchor_body(source, 'MAIN_LOOP'))
        self.assertTrue(code_segment_has_k_tile_dependency('int gk=(bkIdx + 1) * BK + local_k;', 'bkIdx'))
        self.assertFalse(code_segment_has_k_tile_dependency('int gk=BK + local_k;', 'bkIdx'))
    def test_current_parent_complete_source_not_scoped_or_initial(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = '// parent-specific helper outside every anchor\nvoid previous_step() {}'
            (root/'cuda_kernel.cuh').write_text(source)
            (root/'main.cpp').write_text('BENCHMARK_NOT_FOR_LLM')
            context = load_strategy_scoped_code_context({}, root,
                ['cuda_kernel.cuh', 'main.cpp'], {'strategy_id':'Vectorization.GlobalLoadAB.float4'})
            self.assertEqual(context['files']['cuda_kernel.cuh'], source)
            self.assertNotIn('main.cpp', context['files'])

    def test_fixed_cases_repaired_idempotently_without_optimization_removal(self):
        fixtures = Path(__file__).parent/'fixtures/repair_20260916'
        for case in sorted(fixtures.glob('case*')):
            with self.subTest(case=case.name):
                source = (case/'cuda_kernel.cuh').read_text(encoding='utf-8')
                ir = load_json(case/'failed_ir.json')
                repaired, changes = repair_cooperative_loads(source, ir)
                self.assertTrue(changes)
                self.assertEqual(repair_cooperative_loads(repaired, ir), (repaired, []))
                self.assertEqual(preservation_defects(source, repaired, {'strategy_id':'Repair.PreserveAppliedStrategies'}), [])
                self.assertEqual(memory_access_consistency_defects(repaired), [])
                unsupported = {**ir, 'problem': {**ir['problem'], 'K': 19}}
                self.assertEqual(repair_cooperative_loads(source, unsupported), (source, []))


if __name__ == '__main__':
    unittest.main()
