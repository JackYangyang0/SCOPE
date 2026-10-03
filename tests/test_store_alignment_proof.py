import unittest
from SCOPE.verification.store_alignment_proof import prove_fragment_store_alignment


class StoreAlignmentProofTests(unittest.TestCase):
    def test_direct_offset_and_shifted_negative(self):
        harness = 'cudaMalloc(&d_C, bytes_C); cuda_gemm(M,N,K,alpha,d_A,d_B,beta,d_C);'
        source = self.source().replace('FLOAT4(C[c_index])', 'FLOAT4(C[OFFSET(global_m, global_n, N)])')
        self.assertIsNotNone(prove_fragment_store_alignment(source, harness, {'problem': {'N': 1024}}))
        self.assertIsNone(prove_fragment_store_alignment(source.replace('OFFSET(global_m, global_n, N)])', 'OFFSET(global_m, global_n + 1, N)])'), harness, {'problem': {'N': 1024}}))
    def source(self):
        return '''#define OFFSET(row, col, ld) ((row) * (ld) + (col))
        const int tile_n0 = blockIdx.x * BN;
        const int global_n = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN;
        const int c_index = OFFSET(global_m, global_n, N);
        FLOAT4(C[c_index]) = out;
        /* LAUNCH_CONFIG_BEGIN */
        static const int BN = 128;
        static const int WN = 32;
        static const int WNITER = 32;
        static const int TN = 4;
        /* LAUNCH_CONFIG_END */'''

    def test_only_aligned_shape_and_harness(self):
        harness = 'cudaMalloc(&d_C, bytes_C); cuda_gemm(M,N,K,alpha,d_A,d_B,beta,d_C);'
        self.assertIsNotNone(prove_fragment_store_alignment(self.source(), harness, {'problem': {'N': 1024}}))
        for source, main, n in [(self.source(), harness, 1025), (self.source(), '', 1024),
                                (self.source().replace('Tcol * TN;', 'Tcol * TN + 1;'), harness, 1024),
                                (self.source().replace('TN = 4', 'TN = 2'), harness, 1024)]:
            self.assertIsNone(prove_fragment_store_alignment(source, main, {'problem': {'N': n}}))


if __name__ == '__main__':
    unittest.main()
