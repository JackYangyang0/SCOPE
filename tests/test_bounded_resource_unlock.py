import unittest
from SCOPE.verification.bounded_resource_unlock import launch_bounds_source, split_k_source

SOURCE='''
__global__ void gemm(int M,int N,int K,float alpha,float *A,float *B,float beta,float *C) {
const int k_tiles = K / BK;
C[c_index] = alpha * results[i][j] + beta * C[c_index];
}
void cuda_gemm(int M,int N,int K,float alpha,float *A,float *B,float beta,float *C) {
static const int BM=128;
static const int BN=128;
static const int BK=16;
static const int WM=16;
static const int WN=64;
gemm<BM,BN,BK><<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);
}
'''
IR={'problem':{'M':1024,'N':1024,'K':1024,'layout_A':'row_major','layout_B':'row_major','trans_A':False,'trans_B':False}}


class BoundedUnlockTests(unittest.TestCase):
    def test_launch_bounds_uses_actual_threads(self):
        self.assertIn('__launch_bounds__(512, 2)',launch_bounds_source(SOURCE,2))

    def test_split_maintains_leading_dimension_and_times_full_call(self):
        result=split_k_source(SOURCE,IR,3)
        self.assertIn('A += scope_first * BK;',result)
        self.assertIn('B += scope_first * BK * N;',result)
        self.assertIn('const int k_tiles = scope_count;',result)
        self.assertIn('cudaMallocAsync',result)
        self.assertIn('cudaFreeAsync',result)
        self.assertIn('scope_reduce<<<',result)
        self.assertIn('beta == 0.0f ? 0.0f',result)
        self.assertEqual(result.count('void cuda_gemm('),1)

    def test_uneven_slices_cover_whole_k_tiles_once(self):
        for slices in (2,3,4,6):
            covered=[]
            for s in range(slices):
                covered.extend(range(64*s//slices,64*(s+1)//slices))
            self.assertEqual(covered,list(range(64)))

    def test_unsupported_kernel_not_replaced(self):
        with self.assertRaises(ValueError):
            split_k_source(SOURCE+' int valid_k;',IR,2)
        with self.assertRaises(ValueError):
            split_k_source(SOURCE,{**IR,'problem':{**IR['problem'],'K':1025}},2)

    def test_beta_zero_fast_path_can_seed_workspace_split_k(self):
        source = SOURCE.replace(
            'C[c_index] = alpha * results[i][j] + beta * C[c_index];',
            'C[c_index] = alpha * results[i][j];',
        )
        ir = {
            **IR,
            'epilogue': {'mode': 'beta_zero_fast_path'},
            'strategy': {'applied_strategy_ids': ['Epilogue.BetaZero.FastPath']},
        }
        result = split_k_source(source, ir, 2)
        self.assertIn('scope_reduce<<<', result)


if __name__=='__main__':
    unittest.main()
