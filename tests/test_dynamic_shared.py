import unittest
from SCOPE.specialization.dynamic_shared import normalize_dynamic_shared
from SCOPE.specialization.padding_variants import padding_variant


class DynamicSharedTests(unittest.TestCase):
    def source(self):
        return '''extern __shared__ __align__(16) unsigned char scope_shared[];
float (&As)[2][BK][BM] = *reinterpret_cast<float (*)[2][BK][BM]>(scope_shared);
float (&Bs)[2][BK][BN] = *reinterpret_cast<float (*)[2][BK][BN]>(scope_shared + 2 * BK * BM * sizeof(float));
As[s][k][m] = x; Bs[s][k][n] = y;
cudaFuncSetAttribute(gemm<BM, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
gemm<BM, BN><<<grid, block,65536>>>(A, B, C);'''

    def test_size_is_symbolic_and_idempotent(self):
        code, ok = normalize_dynamic_shared(self.source())
        self.assertTrue(ok)
        self.assertNotIn('65536', code)
        self.assertEqual(normalize_dynamic_shared(code)[0], code)

    def test_padded_offsets_and_launch_size(self):
        for mode, pads in [('none', 0), ('a_only', 1), ('ab', 2)]:
            code, metadata, reason = padding_variant(self.source(), {'BM': 128, 'BN': 128, 'BK': 16}, mode)
            self.assertIsNone(reason)
            self.assertEqual(metadata['shared_memory_bytes'], 128 * (256 + pads))
            self.assertNotIn('65536', code)
            if mode != 'none':
                self.assertIn('scope_shared + 2 * BK * (BM + 1) * sizeof(float)', code)

    def test_unsafe_vector_and_offset_rejected(self):
        tile = {'BM': 128, 'BN': 128, 'BK': 16}
        self.assertIsNotNone(padding_variant(self.source() + '\nFLOAT4(Bs[s][k][n]) = v;', tile, 'ab')[2])
        self.assertFalse(normalize_dynamic_shared(self.source().replace('scope_shared + 2', 'scope_shared + 3'))[1])
