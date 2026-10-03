import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.verification.memory_access_plan import enforce_memory_access_plan
from SCOPE.verification.optimization_preservation import preservation_defects, vector_load_widths
from SCOPE.verification.strategy_realization_oracle import verify_strategy_realization
from SCOPE.verification.locked_repair import repair_evidence


SOURCE = '''
__shared__ float As[2][BK][BM];
__shared__ float Bs[2][BK][BN];
const int a_g_m = tile_m0 + a_m;
const int A_offset = tile_m0 * K;
float4 av = *reinterpret_cast<const float4*>(A + a_g_m * K + k + A_offset);
float4 bv = *reinterpret_cast<const float4*>(B + offset);
'''


class StrategyPreservingMemoryCheckTests(unittest.TestCase):
    def test_default_check_does_not_rewrite_defective_vector_kernel(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'cuda_kernel.cuh'
            path.write_text(SOURCE)
            before = path.read_bytes()
            report = enforce_memory_access_plan(root)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report['status'], 'fail')
            self.assertFalse(report['materialized'])
            self.assertTrue(report['defects'])
            self.assertEqual(report['mode'], 'diagnostic_only_preserve_strategies')

    def test_pointer_arithmetic_and_macro_reads_are_equivalent(self):
        macro = 'float4 av=FLOAT4(A[i]); float4 bv=FLOAT4(B[j]);'
        self.assertEqual(vector_load_widths(SOURCE), vector_load_widths(macro))
        self.assertEqual(preservation_defects(macro, SOURCE, {'strategy_id':'Repair.PreserveAppliedStrategies'}), [])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'cuda_kernel.cuh').write_text(SOURCE)
            report=verify_strategy_realization(root, ['Vectorization.GlobalLoadAB.float4'], {})
            self.assertTrue(report['hard_gate_passed'])

    def test_declarations_comments_and_one_operand_do_not_satisfy_ab(self):
        false_evidence = '#define FLOAT4(x) whatever(x)\n// float4 x=FLOAT4(B[i]);\nfloat4 *p=reinterpret_cast<float4*>(B + offset);'
        self.assertEqual(vector_load_widths(false_evidence), {})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for source in (false_evidence, 'float4 av=FLOAT4(A[i]); float2 bv=FLOAT2(B[j]);'):
                (root/'cuda_kernel.cuh').write_text(source)
                self.assertFalse(verify_strategy_realization(root, ['Vectorization.GlobalLoadAB.float4'], {})['hard_gate_passed'])

    def test_repair_cannot_reduce_float4_to_float2(self):
        after=SOURCE.replace('float4', 'float2')
        defects=preservation_defects(SOURCE, after, {'strategy_id':'Repair.PreserveAppliedStrategies'})
        self.assertEqual(len(defects), 2)
        self.assertTrue(all('width reduced' in defect for defect in defects))
        self.assertEqual(preservation_defects(SOURCE, after, {'strategy_id':'Vectorization.GlobalLoadAB.float2'}), [])

    def test_memory_diagnostics_reach_repair_context(self):
        diagnostic={'status':'fail','defects':[{'id':'A_BLOCK_OFFSET_APPLIED_TWICE'}]}
        evidence=repair_evidence({'memory_access_plan_verification':diagnostic}, {}, SOURCE, SOURCE)
        self.assertEqual(evidence['memory_access_diagnostics'], diagnostic)

    def test_terminal_check_still_compiles_original_source_after_static_failure(self):
        from SCOPE.app import verify_terminal_chain_once
        class ReachedCompiler(Exception):
            pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'cuda_kernel.cuh'
            source.write_text(SOURCE)
            before=source.read_bytes()
            ir={'memory':{'access_plan':{'mode':'selected_vector_load'}}, 'strategy':{'applied_strategy_ids':['Vectorization.GlobalLoadAB.float4']}}
            original_ir=copy.deepcopy(ir)
            def compile_probe(*args, **kwargs):
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(kwargs['ir']['memory'], original_ir['memory'])
                self.assertFalse(kwargs['ir']['memory_access_plan_verification']['materialized'])
                raise ReachedCompiler()
            with patch('SCOPE.app.check_after_codegen', return_value={}), \
                 patch('SCOPE.app.extract_code_ast', return_value={}), \
                 patch('SCOPE.app.check_terminal_code_completeness', return_value={}), \
                 patch('SCOPE.app.verify_build_and_run', side_effect=compile_probe):
                with self.assertRaises(ReachedCompiler):
                    verify_terminal_chain_once(ir, root, 'test', {}, 1)
            self.assertEqual(ir, original_ir)


if __name__ == '__main__':
    unittest.main()
