import unittest
from SCOPE.specialization.shape_search import specialize_seed, TILE_KEYS


class TileMetadataTests(unittest.TestCase):
    def test_only_generator_metadata_assertion_changes(self):
        source = 'template<int BM> void gemm() {}\n'
        source += 'static_assert(BM == 64 && BN == 128 && BK == 16, "SCOPE launch config and shared-memory IR disagree");\n'
        source += 'static_assert(BK % 4 == 0, "vector alignment");\n'
        source += '/* LAUNCH_CONFIG_BEGIN */\n'
        source += '\n'.join(f'static const int {k} = 16;' for k in TILE_KEYS)
        source += '\n/* LAUNCH_CONFIG_END */'
        tile = dict.fromkeys(TILE_KEYS, 32)
        result = specialize_seed({'sources': {'cuda_kernel.cuh': source}}, tile)
        self.assertIn('BM == 32 && BN == 32 && BK == 32', result)
        self.assertIn('static_assert(BK % 4 == 0, "vector alignment");', result)
        self.assertIn('static const int BM = 32;', result)
        self.assertIn('BM == 64', source)

    def test_legacy_scope_metadata_assertion_changes(self):
        source = 'template<int BM> void gemm() {}\n'
        source += 'static_assert(BM == 64 && BN == 128 && BK == 16, "SCOPE launch config and shared-memory IR disagree");\n'
        source += '/* LAUNCH_CONFIG_BEGIN */\n'
        source += '\n'.join(f'static const int {k} = 16;' for k in TILE_KEYS)
        source += '\n/* LAUNCH_CONFIG_END */'
        tile = dict.fromkeys(TILE_KEYS, 32)
        result = specialize_seed({'sources': {'cuda_kernel.cuh': source}}, tile)
        self.assertIn('BM == 32 && BN == 32 && BK == 32', result)


if __name__ == '__main__':
    unittest.main()
