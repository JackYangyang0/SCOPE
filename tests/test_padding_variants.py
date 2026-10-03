import unittest
from SCOPE.specialization.padding_variants import padding_variant
from SCOPE.specialization.shape_search import specialize_seed, TILE_KEYS


class PaddingVariantsTests(unittest.TestCase):
    def test_direct_layout_and_resource_updates(self):
        source = '__shared__ float As[2][BK][BM + 1];\n__shared__ float Bs[2][BK][BN + 1];\nAs[s][k][m] = x; Bs[s][k][n] = y;'
        tile = {'BM': 128, 'BN': 128, 'BK': 16}
        for mode, pads in [('none', (0, 0)), ('a_only', (1, 0)), ('ab', (1, 1))]:
            code, metadata, reason = padding_variant(source, tile, mode)
            self.assertIsNone(reason)
            self.assertIn('As[s][k][m] = x;', code)
            self.assertEqual(metadata['shared_A']['padding'], pads[0])
            self.assertEqual(metadata['shared_B']['padding'], pads[1])
            self.assertEqual(metadata['shared_memory_bytes'], 4 * 2 * 16 * (256 + sum(pads)))

    def test_unsafe_shared_representations_are_skipped(self):
        source = '__shared__ float As[2][BK][BM];\n__shared__ float Bs[2][BK][BN];\n'
        for use in ['FLOAT4(Bs[s][k][n]) = v;', 'auto p = &Bs[0][0];', 'cp.async;', 'auto p = reinterpret_cast<float*>(Bs);']:
            code, _, reason = padding_variant(source + use, {'BM': 64, 'BN': 64, 'BK': 16}, 'ab')
            self.assertIsNone(code)
            self.assertTrue(reason)

    def test_generated_shape_locks_follow_new_tile(self):
        source = 'template<int BM> void gemm() {}\n/* LAUNCH_CONFIG_BEGIN */\n'
        source += '\n'.join(f'static const int {k} = 16;' for k in TILE_KEYS)
        source += '\nstatic_assert(BM == 16 && TN == 16, "Compiler.TemplateSpecialization.ShapeStatic: launch config must match specialized shape");\n/* LAUNCH_CONFIG_END */\n'
        source += 'static_assert((BM * BN) / (WM * WN) * 32 == 512, "Compiler.TemplateSpecialization.ShapeStatic: threads_per_block must be 512");\n'
        source += 'static_assert(TN == 4, "algorithm requires four values");'
        tile = dict.fromkeys(TILE_KEYS, 32)
        result = specialize_seed({'sources': {'cuda_kernel.cuh': source}}, tile)
        self.assertIn('BM == 32 && TN == 32', result)
        self.assertIn('threads_per_block must be 32', result)
        self.assertIn('TN == 4, "algorithm requires four values"', result)


if __name__ == '__main__':
    unittest.main()
