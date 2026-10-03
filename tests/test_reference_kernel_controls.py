import unittest

from SCOPE.compare_reference_kernel import vector_shared_b, constant_load_stride, static_store, fixed_load_iterations
from SCOPE.llm.strategy_examples import strategy_with_examples


class ReferenceKernelControlsTests(unittest.TestCase):
    def source(self):
        groups = '\n'.join('\n'.join(f'Bs[{stage}][bk][bn + {i}] = bv.{c};'
                                     for i, c in enumerate('xyzw')) for stage in ('0', 'mem_flag'))
        return ('static const int BN = 64;\n__shared__ float Bs[2][BK][BN];\n'
                'const int bn = (v % b_vec_per_row) * 4;\n' + groups)

    def test_both_load_phases_vectorized(self):
        updated = vector_shared_b(self.source())
        self.assertIn('FLOAT4(Bs[0][bk][bn]) = bv;', updated)
        self.assertIn('FLOAT4(Bs[mem_flag][bk][bn]) = bv;', updated)
        self.assertNotIn('bv.x', updated)

    def test_padded_layout_not_vector_cast(self):
        with self.assertRaises(ValueError):
            vector_shared_b(self.source().replace('[BK][BN]', '[BK][BN + 1]'))

    def test_contract_injected_without_mutating_strategy(self):
        strategy = {'strategy_id': 'Vectorization.GlobalLoadAB.float4'}
        expanded = strategy_with_examples(strategy, {'examples': {}, 'strategy_bindings': {}})
        self.assertIn('coupling_contracts', expanded)
        self.assertNotIn('coupling_contracts', strategy)

    def test_constant_stride_requires_matching_launch(self):
        with self.assertRaises(ValueError):
            constant_load_stride('const int threads = blockDim.x * blockDim.y * blockDim.z;')
        code = 'const int threads = blockDim.x * blockDim.y * blockDim.z;\ndim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);'
        self.assertIn('const int threads = (BM * BN)', constant_load_stride(code))

    def test_guard_removal_requires_divisible_shape(self):
        code = 'static const int BM = 64; static const int BN = 128; static const int BK = 16; if (global_m < M && global_n < N) {}'
        self.assertIn('if (true)', static_store(code, 512))
        with self.assertRaises(ValueError):
            static_store(code, 513)

    def load_source(self, bk=16):
        tile = dict(BM=64, BN=128, BK=bk, WM=32, WN=32, TM=4, TN=4, WMITER=16, WNITER=32)
        return ('\n'.join(f'static const int {key} = {value};' for key, value in tile.items())
                + '\nconst int threads = blockDim.x * blockDim.y * blockDim.z;'
                + '\ndim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);\n'
                + '\n'.join('for (int v = linear_tid; v < ' + name
                            + '_vec_count; v += threads) { consume(v); }'
                            for name in ('a', 'b', 'a', 'b')))

    def test_fixed_load_iterations_cover_both_buffers(self):
        result = fixed_load_iterations(self.load_source())
        self.assertEqual(result.count('const int v = linear_tid + offset;'), 4)
        self.assertNotIn('v += threads', result)

    def test_fixed_load_iterations_reject_partial_groups(self):
        with self.assertRaises(ValueError):
            fixed_load_iterations(self.load_source(bk=8))

    def test_fixed_load_iterations_reject_missing_phase(self):
        with self.assertRaises(ValueError):
            fixed_load_iterations(self.load_source().replace('a_vec_count', 'unknown_count'))


if __name__ == '__main__':
    unittest.main()
