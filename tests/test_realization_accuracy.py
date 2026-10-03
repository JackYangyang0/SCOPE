import unittest
from pathlib import Path

from SCOPE.verification.gemm_semantic_checker import check_register_accumulator_coverage
from SCOPE.verification.strategy_realization_oracle import _check_strategy
from SCOPE.verification.memory_access_plan import vector_mapping_is_proven


class RealizationAccuracyTests(unittest.TestCase):
    def test_next_tile_direct_vector_addresses(self):
        source = '''/* NEXT_TILE_LOAD_BEGIN */
        const int next_k_base = (bkIdx + 1) * BK;
        const int ak = (v % (BK / 4)) * 4;
        const float4 av = FLOAT4(A[(tile_m0 + am) * K + next_k_base + ak]);
        const int bn = (v % (BN / 4)) * 4;
        const float4 bv = FLOAT4(B[(next_k_base + bk) * N + tile_n0 + bn]);
        /* NEXT_TILE_LOAD_END */'''
        self.assertTrue(vector_mapping_is_proven(source))
        self.assertFalse(vector_mapping_is_proven(source.replace('+ ak]', '+ ak + 1]')))
        self.assertFalse(vector_mapping_is_proven(source.replace('* BK;', '* BK + 1;')))

    def test_fma_accumulation_is_recognized(self):
        code = 'float results[TM][TN] = {0.0f}; results[i][j] = fmaf(a[i], b[j], results[i][j]);'
        self.assertEqual(check_register_accumulator_coverage(code, 2, 4)['status'], 'pass')

    def test_fma_overwrite_is_not_accumulation(self):
        for rhs in ('0.0f', 'results[j][i]'):
            code = 'float results[TM][TN] = {0.0f}; results[i][j] = fmaf(a[i], b[j], ' + rhs + ');'
            self.assertEqual(check_register_accumulator_coverage(code, 2, 4)['status'], 'fail')

    def test_shared_declaration_does_not_prove_transpose(self):
        report = _check_strategy('Layout.SharedMemory.TransposeA', '__shared__ float As[2][BM][BK];', '')
        self.assertEqual(report['status'], 'not_realized')
        self.assertTrue(report['critical'])

    def test_transposed_declaration_is_only_partial_evidence(self):
        report = _check_strategy('Layout.SharedMemory.TransposeA', '__shared__ float As[2][BK][BM];', '')
        self.assertEqual(report['status'], 'partially_realized')

    def test_unknown_layout_is_not_rejected_by_name(self):
        report = _check_strategy('Layout.SharedMemory.TransposeA', '__shared__ float custom[8192];', '')
        self.assertEqual(report['status'], 'unknown')

    def test_double_buffers_do_not_prove_overlap(self):
        report = _check_strategy('Pipeline.DoubleBuffer.SharedAB',
                                 '__shared__ float As[2][BM][BK]; __shared__ float Bs[2][BK][BN];', '')
        self.assertEqual(report['status'], 'partially_realized')
        self.assertEqual(report['overlap_status'], 'unproven')


if __name__ == '__main__':
    unittest.main()
