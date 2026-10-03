import unittest

from SCOPE.verification.strategy_realization_oracle import _check_strategy
from SCOPE.verification.pipeline_evidence import pipeline_evidence
from SCOPE.verification.locked_repair import vector_store_contract


class UnlockRealizationRegressions(unittest.TestCase):
    def test_padding_absent_is_not_realized(self):
        report = _check_strategy('Layout.SharedMemory.PaddingA.Plus1',
                                 '__shared__ float As[2][BM][BK];', '')
        self.assertEqual(report['status'], 'not_realized')

    def test_padding_declaration_only_is_partial(self):
        report = _check_strategy('Layout.SharedMemory.PaddingA.Plus1',
                                 '__shared__ float As[2][BK][BM + 1];', '')
        self.assertEqual(report['status'], 'partially_realized')

    def test_unrecognized_padding_is_unknown(self):
        report = _check_strategy('Layout.SharedMemory.PaddingA.Plus1',
                                 '__shared__ float custom[8192];', '')
        self.assertEqual(report['status'], 'unknown')

    def test_read_of_c_is_not_a_store(self):
        for sid in ('Vectorization.StoreC.float4', 'Epilogue.StoreC.Vectorized.float4'):
            report = _check_strategy(sid, 'float4 old = FLOAT4(C[index]);', '')
            self.assertNotEqual(report['status'], 'realized')
            self.assertTrue(report['critical'])

    def test_both_store_labels_recognize_write(self):
        for sid in ('Vectorization.StoreC.float4', 'Epilogue.StoreC.Vectorized.float4'):
            for source in ('FLOAT4(C[index]) = out;', '*reinterpret_cast<float4*>(&C[index]) = out;'):
                self.assertEqual(_check_strategy(sid, source, '')['status'], 'realized')

    def test_serialized_compute_load_is_reported(self):
        source = '''for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
        for (int k = 0; k < BK; ++k) { results[0] += As[comp_flag][k][0]; }
        As[mem_flag][0][0] = A[0]; __syncthreads(); }'''
        report = pipeline_evidence(source)
        self.assertTrue(report['compute_load_barrier_pattern'])
        self.assertFalse(report['overlap_proven'])

    def test_store_repair_has_ownership_and_bounds_contract(self):
        report = vector_store_contract('', {})
        obligations = ' '.join(report['obligations'])
        for required in ('TN=2', 'global_n + 3 < N', 'locked', 'accumulator'):
            self.assertIn(required, obligations)


if __name__ == '__main__':
    unittest.main()
